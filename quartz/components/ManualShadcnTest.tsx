import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import { Button } from "@/components/ui/button"

const ManualShadcnTest: QuartzComponent = ({ displayClass }: QuartzComponentProps) => {
  return (
    <div className={`manual-shadcn-test ${displayClass ?? ""}`}>
      <h2>Manual Shadcn Button Test</h2>
      <div className="space-y-4 p-4">
        <Button>Default Button</Button>
        <Button variant="destructive">Destructive Button</Button>
        <Button variant="outline">Outline Button</Button>
        <Button variant="secondary">Secondary Button</Button>
        <Button variant="ghost">Ghost Button</Button>
        <Button variant="link">Link Button</Button>
        <div className="space-x-2">
          <Button size="sm">Small</Button>
          <Button size="default">Default</Button>
          <Button size="lg">Large</Button>
        </div>
      </div>
    </div>
  )
}

ManualShadcnTest.css = `
.manual-shadcn-test {
  margin: 1rem 0;
  padding: 1rem;
  border: 1px solid var(--border);
  border-radius: 0.5rem;
}

.manual-shadcn-test .space-y-4 > * + * {
  margin-top: 1rem;
}

.manual-shadcn-test .space-x-2 > * + * {
  margin-left: 0.5rem;
}
`

export default (() => ManualShadcnTest) satisfies QuartzComponentConstructor 