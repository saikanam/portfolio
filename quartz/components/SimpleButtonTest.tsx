import React from "react"
import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"

const SimpleButtonTest: QuartzComponent = ({ displayClass }: QuartzComponentProps) => {
  return (
    <div className={`simple-button-test p-4 bg-gray-100 dark:bg-gray-800 rounded-lg ${displayClass ?? ""}`}>
      <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-gray-100">Simple Button Test</h3>
      <div className="flex flex-wrap gap-2 mb-4">
        <button className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition-colors">
          Basic Button
        </button>
        <button className="px-4 py-2 bg-gray-200 text-gray-800 rounded hover:bg-gray-300 transition-colors">
          Secondary Button
        </button>
        <button className="px-4 py-2 border border-gray-300 bg-white text-gray-700 rounded hover:bg-gray-50 transition-colors">
          Outline Button
        </button>
      </div>
      <p className="text-sm text-gray-600 dark:text-gray-400">
        If you can see these styled buttons, React + Tailwind is working!
      </p>
    </div>
  )
}

SimpleButtonTest.displayName = "SimpleButtonTest"

export default (() => SimpleButtonTest) satisfies QuartzComponentConstructor 