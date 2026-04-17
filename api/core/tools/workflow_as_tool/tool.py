from __future__ import annotations

import json
import logging
from collections.abc import Generator, Mapping, Sequence
from typing import Any, cast

from graphon.file import FILE_MODEL_IDENTITY, File, FileTransferMethod
from graphon.model_runtime.entities.llm_entities import LLMUsage, LLMUsageMetadata
from sqlalchemy import select

from core.app.file_access import DatabaseFileAccessController
from core.db.session_factory import session_factory
from core.tools.__base.tool import Tool
from core.tools.__base.tool_runtime import ToolRuntime
from core.tools.entities.tool_entities import (
    ToolEntity,
    ToolInvokeMessage,
    ToolParameter,
    ToolProviderType,
)
from core.tools.errors import ToolInvokeError
from core.workflow.file_reference import resolve_file_record_id
from factories.file_factory import build_from_mapping
from models import Account, Tenant
from models.model import App, EndUser
from models.utils.file_input_compat import build_file_from_stored_mapping
from models.workflow import Workflow

logger = logging.getLogger(__name__)
_file_access_controller = DatabaseFileAccessController()


class WorkflowTool(Tool):
    """
    Workflow tool.
    """

    def __init__(
        self,
        workflow_app_id: str,
        workflow_as_tool_id: str,
        version: str,
        workflow_entities: dict[str, Any],
        workflow_call_depth: int,
        entity: ToolEntity,
        runtime: ToolRuntime,
        label: str = "Workflow",
    ):
        self.workflow_app_id = workflow_app_id
        self.workflow_as_tool_id = workflow_as_tool_id
        self.version = version
        self.workflow_entities = workflow_entities
        self.workflow_call_depth = workflow_call_depth
        self.label = label
        self._latest_usage = LLMUsage.empty_usage()

        super().__init__(entity=entity, runtime=runtime)

    def tool_provider_type(self) -> ToolProviderType:
        """
        get the tool provider type

        :return: the tool provider type
        """
        return ToolProviderType.WORKFLOW

    def _invoke(
        self,
        user_id: str,
        tool_parameters: dict[str, Any],
        conversation_id: str | None = None,
        app_id: str | None = None,
        message_id: str | None = None,
    ) -> Generator[ToolInvokeMessage, None, None]:
        """
        invoke the tool
        """
        app = self._get_app(app_id=self.workflow_app_id)
        workflow = self._get_workflow(app_id=self.workflow_app_id, version=self.version)

        # Try to extract pre-resolved file objects from tool_parameters
        # BEFORE _transform_args, so we can bypass the DB re-lookup in the
        # generator that may fail under restrictive file-access scopes.
        pre_built_files = self._extract_pre_resolved_files(tool_parameters)

        # transform the tool parameters
        tool_parameters, files = self._transform_args(tool_parameters=tool_parameters)

        from core.app.apps.workflow.app_generator import WorkflowAppGenerator

        generator = WorkflowAppGenerator()
        assert self.runtime is not None
        assert self.runtime.invoke_from is not None

        user = self._resolve_user(user_id=user_id)
        if user is None:
            raise ToolInvokeError("User not found")

        self._latest_usage = LLMUsage.empty_usage()

        result = generator.generate(
            app_model=app,
            workflow=workflow,
            user=user,
            args={"inputs": tool_parameters, "files": files},
            invoke_from=self.runtime.invoke_from,
            streaming=False,
            call_depth=self.workflow_call_depth + 1,
            # NOTE(QuantumGhost): We explicitly set `pause_state_config` to `None`
            # because workflow pausing mechanisms (such as HumanInput) are not
            # supported within WorkflowTool execution context.
            pause_state_config=None,
            system_files=pre_built_files if pre_built_files else None,
        )
        assert isinstance(result, dict)
        data = result.get("data", {})

        if err := data.get("error"):
            raise ToolInvokeError(err)

        outputs = data.get("outputs")
        if outputs is None:
            outputs = {}
        else:
            outputs, files = self._extract_files(outputs)  # type: ignore
            for file in files:
                yield self.create_file_message(file)  # type: ignore

        # traverse `outputs` field and create variable messages
        for key, value in outputs.items():
            if key not in {"text", "json", "files"}:
                yield self.create_variable_message(variable_name=key, variable_value=value)

        self._latest_usage = self._derive_usage_from_result(data)

        yield self.create_text_message(json.dumps(outputs, ensure_ascii=False))
        yield self.create_json_message(outputs, suppress_output=True)

    @property
    def latest_usage(self) -> LLMUsage:
        return self._latest_usage

    @classmethod
    def _derive_usage_from_result(cls, data: Mapping[str, Any]) -> LLMUsage:
        usage_dict = cls._extract_usage_dict(data)
        if usage_dict is not None:
            return LLMUsage.from_metadata(cast(LLMUsageMetadata, dict(usage_dict)))

        total_tokens = data.get("total_tokens")
        total_price = data.get("total_price")
        if total_tokens is None and total_price is None:
            return LLMUsage.empty_usage()

        usage_metadata: dict[str, Any] = {}
        if total_tokens is not None:
            try:
                usage_metadata["total_tokens"] = int(str(total_tokens))
            except (TypeError, ValueError):
                pass
        if total_price is not None:
            usage_metadata["total_price"] = str(total_price)
        currency = data.get("currency")
        if currency is not None:
            usage_metadata["currency"] = currency

        if not usage_metadata:
            return LLMUsage.empty_usage()

        return LLMUsage.from_metadata(cast(LLMUsageMetadata, usage_metadata))

    @classmethod
    def _extract_usage_dict(cls, payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
        usage_candidate = payload.get("usage")
        if isinstance(usage_candidate, Mapping):
            return usage_candidate

        metadata_candidate = payload.get("metadata")
        if isinstance(metadata_candidate, Mapping):
            usage_candidate = metadata_candidate.get("usage")
            if isinstance(usage_candidate, Mapping):
                return usage_candidate

        for value in payload.values():
            if isinstance(value, Mapping):
                found = cls._extract_usage_dict(value)
                if found is not None:
                    return found
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                for item in value:
                    if isinstance(item, Mapping):
                        found = cls._extract_usage_dict(item)
                        if found is not None:
                            return found
        return None

    def fork_tool_runtime(self, runtime: ToolRuntime) -> WorkflowTool:
        """
        fork a new tool with metadata

        :return: the new tool
        """
        return self.__class__(
            entity=self.entity.model_copy(),
            runtime=runtime,
            workflow_app_id=self.workflow_app_id,
            workflow_as_tool_id=self.workflow_as_tool_id,
            workflow_entities=self.workflow_entities,
            workflow_call_depth=self.workflow_call_depth,
            version=self.version,
            label=self.label,
        )

    def _extract_pre_resolved_files(self, tool_parameters: dict[str, Any]) -> list[File] | None:
        """Extract pre-resolved File objects from SYSTEM_FILES parameters.

        When file dicts contain ``upload_file_id`` (set by the agent
        serialisation layer), we can reconstruct File objects directly via
        ``File.model_validate`` and skip the generator's DB round-trip.

        Returns *None* when no pre-resolved files are found so the generator
        falls back to its normal ``build_from_mappings`` path.
        """
        parameter_rules = self.get_merged_runtime_parameters()
        result: list[File] = []
        for parameter in parameter_rules:
            if parameter.type != ToolParameter.ToolParameterType.SYSTEM_FILES:
                continue
            file_list = tool_parameters.get(parameter.name)
            if not file_list or not isinstance(file_list, list):
                continue
            for f in file_list:
                if not isinstance(f, Mapping):
                    continue
                # Only handle dicts that were pre-resolved (have an explicit
                # file id AND the full File metadata from model_dump).
                has_id = any(f.get(k) for k in ("upload_file_id", "tool_file_id", "datasource_file_id"))
                if not has_id:
                    return None  # not pre-resolved; fall back
                try:
                    result.append(File.model_validate(f))
                except Exception:
                    return None
        return result or None

    def _resolve_user(self, user_id: str) -> Account | EndUser | None:
        """
        Resolve user object in both HTTP and worker contexts.

        In HTTP context: dereference the current_user LocalProxy (can return Account or EndUser).
        In worker context: load Account(knowledge pipeline) or EndUser(trigger) from database by user_id.

        Returns:
            Account | EndUser | None: The resolved user object, or None if resolution fails.
        """
        return self._resolve_user_from_database(user_id=user_id)

    def _resolve_user_from_database(self, user_id: str) -> Account | EndUser | None:
        """
        Resolve user from database (worker/Celery context).
        """
        with session_factory.create_session() as session:
            tenant_stmt = select(Tenant).where(Tenant.id == self.runtime.tenant_id)
            tenant = session.scalar(tenant_stmt)
            if not tenant:
                return None

            user_stmt = select(Account).where(Account.id == user_id)
            user = session.scalar(user_stmt)
            if user:
                user.current_tenant = tenant
                session.expunge(user)
                return user

            end_user_stmt = select(EndUser).where(EndUser.id == user_id, EndUser.tenant_id == tenant.id)
            end_user = session.scalar(end_user_stmt)
            if end_user:
                session.expunge(end_user)
                return end_user

            return None

    def _get_workflow(self, app_id: str, version: str) -> Workflow:
        """
        get the workflow by app id and version
        """
        with session_factory.create_session() as session, session.begin():
            if not version:
                stmt = (
                    select(Workflow)
                    .where(Workflow.app_id == app_id, Workflow.version != Workflow.VERSION_DRAFT)
                    .order_by(Workflow.created_at.desc())
                )
                workflow = session.scalars(stmt).first()
            else:
                stmt = select(Workflow).where(Workflow.app_id == app_id, Workflow.version == version)
                workflow = session.scalar(stmt)

            if not workflow:
                raise ValueError("workflow not found or not published")

            session.expunge(workflow)
            return workflow

    def _get_app(self, app_id: str) -> App:
        """
        get the app by app id
        """
        stmt = select(App).where(App.id == app_id)
        with session_factory.create_session() as session, session.begin():
            app = session.scalar(stmt)
            if not app:
                raise ValueError("app not found")

            session.expunge(app)
            return app

    def _transform_args(self, tool_parameters: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, str | None]]]:
        """
        transform the tool parameters

        :param tool_parameters: the tool parameters
        :return: tool_parameters, files
        """
        parameter_rules = self.get_merged_runtime_parameters()
        parameters_result = {}
        files = []
        for parameter in parameter_rules:
            if parameter.type == ToolParameter.ToolParameterType.SYSTEM_FILES:
                file = tool_parameters.get(parameter.name)
                if file:
                    try:
                        for f in file:
                            if not isinstance(f, Mapping):
                                continue
                            # If the mapping already carries a resolved file id
                            # (e.g. from agent runtime_parameters), build the
                            # file dict directly to avoid a DB re-lookup that
                            # may fail under a restrictive file-access scope.
                            file_dict = self._try_build_file_dict_from_resolved(f)
                            if file_dict is not None:
                                files.append(file_dict)
                                continue
                            file_obj = build_file_from_stored_mapping(
                                file_mapping=cast(Mapping[str, Any], f),
                                tenant_id=str(self.runtime.tenant_id),
                            )
                            file_dict = {
                                "transfer_method": file_obj.transfer_method.value,
                                "type": file_obj.type.value,
                            }
                            match file_obj.transfer_method:
                                case FileTransferMethod.TOOL_FILE:
                                    file_dict["tool_file_id"] = resolve_file_record_id(file_obj.reference)
                                case FileTransferMethod.LOCAL_FILE:
                                    file_dict["upload_file_id"] = resolve_file_record_id(file_obj.reference)
                                case FileTransferMethod.DATASOURCE_FILE:
                                    file_dict["datasource_file_id"] = resolve_file_record_id(file_obj.reference)
                                case FileTransferMethod.REMOTE_URL:
                                    file_dict["url"] = file_obj.generate_url()
                            files.append(file_dict)
                    except Exception:
                        logger.exception("Failed to transform file %s", file)
            else:
                parameters_result[parameter.name] = tool_parameters.get(parameter.name)

        return parameters_result, files

    @staticmethod
    def _try_build_file_dict_from_resolved(
        mapping: Mapping[str, Any],
    ) -> dict[str, str | None] | None:
        """Return a file dict directly if *mapping* already contains a resolved
        file id (``upload_file_id``, ``tool_file_id``, etc.).

        This avoids a redundant DB round-trip for files that were pre-resolved
        in the agent runtime-parameters serialisation step.
        """
        for id_key in ("upload_file_id", "tool_file_id", "datasource_file_id"):
            if mapping.get(id_key):
                return {
                    "transfer_method": mapping.get("transfer_method"),
                    "type": mapping.get("type"),
                    id_key: mapping[id_key],
                }
        # Not pre-resolved – fall back to the normal DB lookup path.
        return None

    def _extract_files(self, outputs: dict[str, Any]) -> tuple[dict[str, Any], list[File]]:
        """
        extract files from the result

        :return: the result, files
        """
        files: list[File] = []
        result = {}
        for key, value in outputs.items():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and item.get("dify_model_identity") == FILE_MODEL_IDENTITY:
                        item = self._update_file_mapping(item)
                        file = build_from_mapping(
                            mapping=item,
                            tenant_id=str(self.runtime.tenant_id),
                            access_controller=_file_access_controller,
                        )
                        files.append(file)
            elif isinstance(value, dict) and value.get("dify_model_identity") == FILE_MODEL_IDENTITY:
                value = self._update_file_mapping(value)
                file = build_from_mapping(
                    mapping=value,
                    tenant_id=str(self.runtime.tenant_id),
                    access_controller=_file_access_controller,
                )
                files.append(file)

            result[key] = value

        return result, files

    def _update_file_mapping(self, file_dict: dict[str, Any]) -> dict[str, Any]:
        file_id = resolve_file_record_id(file_dict.get("reference") or file_dict.get("related_id"))
        transfer_method = FileTransferMethod.value_of(file_dict.get("transfer_method"))
        match transfer_method:
            case FileTransferMethod.TOOL_FILE:
                file_dict["tool_file_id"] = file_id
            case FileTransferMethod.LOCAL_FILE:
                file_dict["upload_file_id"] = file_id
            case FileTransferMethod.REMOTE_URL | FileTransferMethod.DATASOURCE_FILE:
                pass
        return file_dict
