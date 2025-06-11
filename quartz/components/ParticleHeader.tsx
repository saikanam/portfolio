import React, { useRef, useEffect, useState } from "react"
import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"

const ParticleHeader: QuartzComponent = ({ cfg, fileData, displayClass }: QuartzComponentProps) => {
  return (
    <div className={`particle-header relative w-full h-[200px] flex items-center justify-center bg-[#161618] border-b-2 border-cyan-400 ${displayClass ?? ""}`}>
      {/* Visible test content */}
      <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/20 to-orange-500/20"></div>
      
      {/* Main title */}
      <div className="relative z-10 text-center">
        <h1 className="text-4xl md:text-6xl font-bold text-white mb-2">
          <span className="text-cyan-400">Saik</span>{" "}
          <span className="text-orange-400">Anam</span>
        </h1>
        <p className="text-gray-300 text-sm md:text-base">Interactive Portfolio</p>
      </div>

      {/* Bottom text */}
      <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 z-10">
        <p className="font-mono text-gray-400 text-xs text-center">
          <span className="text-gray-300 hover:text-cyan-400 transition-colors duration-300">Saik Anam's</span>{" "}
          <span className="hover:text-orange-400 transition-colors duration-300">Portfolio</span>
          <br />
          <span className="text-xs opacity-60">(particle effects loading...)</span>
        </p>
      </div>
    </div>
  )
}

ParticleHeader.displayName = "ParticleHeader"

ParticleHeader.css = `
.particle-header {
  position: relative;
  overflow: hidden;
  background-color: #161618;
  min-height: 200px;
  border-bottom: 2px solid #00DCFF;
}
`

export default (() => ParticleHeader) satisfies QuartzComponentConstructor 