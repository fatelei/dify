import type { ViewHistoryProps } from './view-history'
import {
  RiPlayLargeLine,
} from '@remixicon/react'
import { memo } from 'react'
import { useTranslation } from 'react-i18next'
import { useToastContext } from '@/app/components/base/toast'
import { cn } from '@/utils/classnames'
import {
  useChecklistBeforePublish,
  useNodesReadOnly,
  useWorkflowRunValidation,
  useWorkflowStartRun,
} from '../hooks'
import Checklist from './checklist'
import RunMode from './run-mode'
import ViewHistory from './view-history'

const PreviewMode = memo(() => {
  const { t } = useTranslation()
  const { notify } = useToastContext()
  const { handleWorkflowStartRunInChatflow } = useWorkflowStartRun()
  const { hasValidationErrors } = useWorkflowRunValidation()
  const { handleCheckBeforePublish } = useChecklistBeforePublish()

  const handleClick = async () => {
    // Gate preview behind the same validations used for Publish
    if (hasValidationErrors) {
      notify({ type: 'error', message: t('panel.checklistTip', { ns: 'workflow' }) })
      return
    }
    const ok = await handleCheckBeforePublish()
    if (!ok)
      return
    handleWorkflowStartRunInChatflow()
  }

  return (
    <div
      className={cn(
        'flex h-7 items-center rounded-md px-2.5 text-[13px] font-medium text-components-button-secondary-accent-text',
        hasValidationErrors ? 'cursor-not-allowed opacity-50' : 'cursor-pointer hover:bg-state-accent-hover',
      )}
      onClick={handleClick}
    >
      <RiPlayLargeLine className="mr-1 h-4 w-4" />
      {t('common.debugAndPreview', { ns: 'workflow' })}
    </div>
  )
})

export type RunAndHistoryProps = {
  showRunButton?: boolean
  runButtonText?: string
  isRunning?: boolean
  showPreviewButton?: boolean
  viewHistoryProps?: ViewHistoryProps
  components?: {
    RunMode?: React.ComponentType<
      {
        text?: string
      }
    >
  }
}
const RunAndHistory = ({
  showRunButton,
  runButtonText,
  showPreviewButton,
  viewHistoryProps,
  components,
}: RunAndHistoryProps) => {
  const { nodesReadOnly } = useNodesReadOnly()
  const { RunMode: CustomRunMode } = components || {}

  return (
    <div className="flex h-8 items-center rounded-lg border-[0.5px] border-components-button-secondary-border bg-components-button-secondary-bg px-0.5 shadow-xs">
      {
        showRunButton && (
          CustomRunMode ? <CustomRunMode text={runButtonText} /> : <RunMode text={runButtonText} />
        )
      }
      {
        showPreviewButton && <PreviewMode />
      }
      <div className="mx-0.5 h-3.5 w-[1px] bg-divider-regular"></div>
      <ViewHistory {...viewHistoryProps} />
      <Checklist disabled={nodesReadOnly} />
    </div>
  )
}

export default memo(RunAndHistory)
