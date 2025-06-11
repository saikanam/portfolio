import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"
import { 
  AlertTriangle, 
  Info, 
  CheckCircle, 
  XCircle, 
  Lightbulb, 
  Zap, 
  Quote,
  HelpCircle,
  AlertCircle,
  Flame,
  Bug,
  Settings
} from "lucide-react"

import { cn } from "@/lib/utils"

const calloutVariants = cva(
  "relative w-full rounded-lg border px-4 py-3 text-sm grid has-[>svg]:grid-cols-[calc(var(--spacing)*4)_1fr] grid-cols-[0_1fr] has-[>svg]:gap-x-3 gap-y-0.5 items-start [&>svg]:size-4 [&>svg]:translate-y-0.5",
  {
    variants: {
      variant: {
        note: "bg-blue-50 border-blue-200 text-blue-900 [&>svg]:text-blue-600 dark:bg-blue-950/30 dark:border-blue-800 dark:text-blue-100 dark:[&>svg]:text-blue-400",
        tip: "bg-green-50 border-green-200 text-green-900 [&>svg]:text-green-600 dark:bg-green-950/30 dark:border-green-800 dark:text-green-100 dark:[&>svg]:text-green-400",
        important: "bg-purple-50 border-purple-200 text-purple-900 [&>svg]:text-purple-600 dark:bg-purple-950/30 dark:border-purple-800 dark:text-purple-100 dark:[&>svg]:text-purple-400",
        warning: "bg-yellow-50 border-yellow-200 text-yellow-900 [&>svg]:text-yellow-600 dark:bg-yellow-950/30 dark:border-yellow-800 dark:text-yellow-100 dark:[&>svg]:text-yellow-400",
        caution: "bg-orange-50 border-orange-200 text-orange-900 [&>svg]:text-orange-600 dark:bg-orange-950/30 dark:border-orange-800 dark:text-orange-100 dark:[&>svg]:text-orange-400",
        danger: "bg-red-50 border-red-200 text-red-900 [&>svg]:text-red-600 dark:bg-red-950/30 dark:border-red-800 dark:text-red-100 dark:[&>svg]:text-red-400",
        info: "bg-cyan-50 border-cyan-200 text-cyan-900 [&>svg]:text-cyan-600 dark:bg-cyan-950/30 dark:border-cyan-800 dark:text-cyan-100 dark:[&>svg]:text-cyan-400",
        success: "bg-emerald-50 border-emerald-200 text-emerald-900 [&>svg]:text-emerald-600 dark:bg-emerald-950/30 dark:border-emerald-800 dark:text-emerald-100 dark:[&>svg]:text-emerald-400",
        question: "bg-indigo-50 border-indigo-200 text-indigo-900 [&>svg]:text-indigo-600 dark:bg-indigo-950/30 dark:border-indigo-800 dark:text-indigo-100 dark:[&>svg]:text-indigo-400",
        quote: "bg-gray-50 border-gray-200 text-gray-900 [&>svg]:text-gray-600 dark:bg-gray-950/30 dark:border-gray-800 dark:text-gray-100 dark:[&>svg]:text-gray-400",
        example: "bg-teal-50 border-teal-200 text-teal-900 [&>svg]:text-teal-600 dark:bg-teal-950/30 dark:border-teal-800 dark:text-teal-100 dark:[&>svg]:text-teal-400",
        bug: "bg-pink-50 border-pink-200 text-pink-900 [&>svg]:text-pink-600 dark:bg-pink-950/30 dark:border-pink-800 dark:text-pink-100 dark:[&>svg]:text-pink-400",
        abstract: "bg-slate-50 border-slate-200 text-slate-900 [&>svg]:text-slate-600 dark:bg-slate-950/30 dark:border-slate-800 dark:text-slate-100 dark:[&>svg]:text-slate-400",
        todo: "bg-blue-50 border-blue-200 text-blue-900 [&>svg]:text-blue-600 dark:bg-blue-950/30 dark:border-blue-800 dark:text-blue-100 dark:[&>svg]:text-blue-400",
      },
    },
    defaultVariants: {
      variant: "note",
    },
  }
)

const calloutIcons = {
  note: Info,
  tip: Lightbulb,
  important: Zap,
  warning: AlertTriangle,
  caution: AlertCircle,
  danger: XCircle,
  info: Info,
  success: CheckCircle,
  question: HelpCircle,
  quote: Quote,
  example: Settings,
  bug: Bug,
  abstract: Settings,
  todo: CheckCircle,
}

interface CalloutProps extends React.ComponentProps<"div">, VariantProps<typeof calloutVariants> {
  title?: string;
  collapsible?: boolean;
  defaultOpen?: boolean;
}

function Callout({
  className,
  variant = "note",
  title,
  collapsible = false,
  defaultOpen = true,
  children,
  ...props
}: CalloutProps) {
  const [isOpen, setIsOpen] = React.useState(defaultOpen);
  const Icon = calloutIcons[variant as keyof typeof calloutIcons] || Info;

  return (
    <div
      role="alert"
      className={cn(calloutVariants({ variant }), className)}
      {...props}
    >
      <Icon className="mt-0.5" />
      <div className="space-y-2">
        {title && (
          <div 
            className={cn(
              "font-medium tracking-tight flex items-center gap-2",
              collapsible && "cursor-pointer select-none"
            )}
            onClick={collapsible ? () => setIsOpen(!isOpen) : undefined}
          >
            {title}
            {collapsible && (
              <span className="text-xs opacity-60">
                {isOpen ? "▼" : "▶"}
              </span>
            )}
          </div>
        )}
        {(!collapsible || isOpen) && (
          <div className="text-sm [&_p]:leading-relaxed [&_p]:mb-2 [&_p:last-child]:mb-0">
            {children}
          </div>
        )}
      </div>
    </div>
  )
}

export { Callout, type CalloutProps } 