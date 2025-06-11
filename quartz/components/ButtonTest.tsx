import React from "react"
import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import { Button } from "./ui/button"

const ButtonTest: QuartzComponent = ({ displayClass }: QuartzComponentProps) => {
  return (
    <div className={`button-test p-4 bg-gray-100 dark:bg-gray-800 rounded-lg ${displayClass ?? ""}`}>
      <h3 className="text-lg font-semibold mb-4">shadcn/ui Button Test</h3>
      <div className="flex flex-wrap gap-2 mb-4">
        <Button>Default Button</Button>
        <Button variant="secondary">Secondary</Button>
        <Button variant="outline">Outline</Button>
        <Button variant="ghost">Ghost</Button>
        <Button variant="destructive">Destructive</Button>
      </div>
      <div className="flex flex-wrap gap-2">
        <Button size="sm">Small</Button>
        <Button size="default">Default</Button>
        <Button size="lg">Large</Button>
      </div>
    </div>
  )
}

ButtonTest.displayName = "ButtonTest"

export default (() => ButtonTest) satisfies QuartzComponentConstructor 