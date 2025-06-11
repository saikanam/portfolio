import { ReactNode } from "react"
import { cn } from "../../lib/utils"
import { QuartzComponent, QuartzComponentProps, QuartzComponentConstructor } from "../types"

export interface BadgeProps {
  className?: string
  children?: ReactNode
}

export default ((opts: BadgeProps = {}) => {
  const BadgeComponent: QuartzComponent = ({ displayClass }: QuartzComponentProps) => {
    const { className, children } = opts
    
    return (
      <div
        className={cn(
          "p-4 border rounded",
          className,
          displayClass
        )}
      >
        {children}
      </div>
    )
  }

  BadgeComponent.css = `
    /* Add your CSS variables here */
  `

  return BadgeComponent
}) satisfies QuartzComponentConstructor<BadgeProps>
