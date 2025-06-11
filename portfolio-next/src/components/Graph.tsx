'use client'

import React, { useEffect, useRef, useState } from 'react'
import { Network } from 'vis-network/standalone'
import { DataSet } from 'vis-data'
import { GraphData, GraphNode, GraphLink } from '@/lib/markdown'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { H2, Muted } from '@/components/ui/typography'
import { Globe, Maximize2 } from 'lucide-react'
import {
  FloatingPanelRoot,
  FloatingPanelTrigger,
  FloatingPanelContent,
  FloatingPanelHeader,
  FloatingPanelBody,
  FloatingPanelFooter,
  FloatingPanelCloseButton,
} from '@/components/ui/floating-panel'

interface GraphProps {
  localData: GraphData
  globalData: GraphData
  currentSlug?: string
  className?: string
}

export default function Graph({ localData, globalData, currentSlug, className = '' }: GraphProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const globalContainerRef = useRef<HTMLDivElement>(null)
  const localPanelContainerRef = useRef<HTMLDivElement>(null)
  const networkRef = useRef<Network | null>(null)
  const globalNetworkRef = useRef<Network | null>(null)
  const localPanelNetworkRef = useRef<Network | null>(null)
  const [globalGraphMounted, setGlobalGraphMounted] = useState(false)
  const [localPanelMounted, setLocalPanelMounted] = useState(false)

  // Create network visualization with full interactivity
  const createNetwork = (container: HTMLDivElement, data: GraphData, isGlobal = false) => {
    const nodes = new DataSet(
      data.nodes.map((node) => ({
        id: node.id,
        label: isGlobal ? (node.label.length > 15 ? node.label.substring(0, 12) + '...' : node.label) : node.label,
        title: `${node.title}\nTags: ${node.tags.join(', ') || 'none'}`,
        color: {
          background: node.isCurrentPage 
            ? 'hsl(221 83% 53%)' // Vibrant blue for current page
            : node.slug.startsWith('Projects/') 
              ? 'hsl(210 50% 65%)' // Slightly brighter project pages
              : 'hsl(210 45% 75%)', // Brighter other pages for better visibility
          border: node.isCurrentPage ? 'hsl(221 83% 35%)' : 'hsl(210 20% 45%)', // Darker borders for contrast
          highlight: {
            background: 'hsl(142 76% 45%)', // Brighter green highlight
            border: 'hsl(142 76% 35%)'
          },
          hover: {
            background: node.isCurrentPage 
              ? 'hsl(221 83% 60%)' 
              : 'hsl(210 50% 70%)',
            border: 'hsl(210 20% 35%)'
          }
        },
        font: {
          color: 'hsl(210 40% 20%)',
          size: isGlobal ? (node.isCurrentPage ? 11 : 9) : (node.isCurrentPage ? 14 : 12),
          strokeWidth: 2,
          strokeColor: 'white'
        },
        shape: 'dot',
        size: isGlobal 
          ? (node.isCurrentPage ? 18 : node.slug.startsWith('Projects/') ? 14 : 10)
          : (node.isCurrentPage ? 28 : node.slug.startsWith('Projects/') ? 22 : 16),
        shadow: {
          enabled: true,
          color: 'rgba(0,0,0,0.2)',
          size: isGlobal ? 3 : 5,
          x: 1,
          y: 1
        },
        // Store original properties for hover effects
        originalColor: {
          background: node.isCurrentPage 
            ? 'hsl(221 83% 53%)' 
            : node.slug.startsWith('Projects/') 
              ? 'hsl(210 50% 65%)' 
              : 'hsl(210 45% 75%)',
          border: node.isCurrentPage ? 'hsl(221 83% 35%)' : 'hsl(210 20% 45%)'
        },
        fadedColor: {
          background: node.isCurrentPage 
            ? 'hsla(221, 83%, 53%, 0.2)' 
            : node.slug.startsWith('Projects/') 
              ? 'hsla(210, 50%, 65%, 0.2)' 
              : 'hsla(210, 45%, 75%, 0.2)',
          border: node.isCurrentPage ? 'hsla(221, 83%, 35%, 0.2)' : 'hsla(210, 20%, 45%, 0.2)'
        }
      }))
    )

    const edges = new DataSet(
      data.edges.map((edge, index) => ({
        id: index,
        from: edge.from,
        to: edge.to,
        color: {
          color: 'hsl(210 15% 60%)',
          highlight: 'hsl(142 76% 55%)'
        },
        width: isGlobal ? 1 : 2,
        smooth: {
          type: 'continuous',
          roundness: 0.3
        },
        shadow: {
          enabled: !isGlobal,
          color: 'rgba(0,0,0,0.1)',
          size: 3,
          x: 1,
          y: 1
        },
        // Store original properties for hover effects
        originalColor: 'hsl(210 15% 60%)',
        fadedColor: 'hsla(210, 15%, 60%, 0.1)'
      }))
    )

    const options = {
      nodes: {
        borderWidth: 2,
        chosen: {
          node: (values: any, id: string, selected: boolean, hovering: boolean) => {
            if (hovering) {
              values.shadow = true
              values.shadowColor = 'rgba(0,0,0,0.3)'
              values.shadowSize = 8
            }
          }
        }
      },
      edges: {
        arrows: {
          to: {
            enabled: true,
            scaleFactor: isGlobal ? 0.4 : 0.6,
            type: 'arrow'
          }
        },
        smooth: {
          enabled: true,
          type: 'continuous',
          roundness: 0.3
        },
        chosen: {
          edge: (values: any, id: string, selected: boolean, hovering: boolean) => {
            if (hovering) {
              values.color = 'hsl(142 76% 55%)'
              values.width = values.width * 1.5
            }
          }
        }
      },
      physics: {
        enabled: true,
        solver: 'forceAtlas2Based',
        forceAtlas2Based: {
          gravitationalConstant: isGlobal ? -35 : -50,
          centralGravity: isGlobal ? 0.008 : 0.01,
          springLength: isGlobal ? 80 : 100,
          springConstant: isGlobal ? 0.12 : 0.08,
          damping: 0.6,
          avoidOverlap: isGlobal ? 0.1 : 0.3
        },
        maxVelocity: isGlobal ? 25 : 30,
        minVelocity: 0.5,
        stabilization: {
          enabled: true,
          iterations: isGlobal ? 300 : 400,
          updateInterval: 25,
          onlyDynamicEdges: false,
          fit: true
        },
        adaptiveTimestep: true
      },
      interaction: {
        hover: true,
        hoverConnectedEdges: true,
        selectConnectedEdges: false,
        tooltipDelay: 150,
        zoomView: true,
        dragView: true,
        dragNodes: true,
        zoomSpeed: isGlobal ? 0.3 : 0.4,
        keyboard: {
          enabled: true,
          speed: { x: 10, y: 10, zoom: 0.02 },
          bindToWindow: false
        },
        multiselect: false,
        navigationButtons: false
      },
      layout: {
        improvedLayout: true,
        clusterThreshold: 150
      }
    }

    const network = new Network(container, { nodes, edges }, options)

    // Enhanced hover highlighting for immediate connections
    let hoveredNodeId: string | null = null

    const highlightConnections = (nodeId: string | null) => {
      if (nodeId === hoveredNodeId) return // Avoid redundant updates
      hoveredNodeId = nodeId

      if (nodeId) {
        // Get connected nodes and edges
        const connectedNodes = network.getConnectedNodes(nodeId) as string[]
        const connectedEdges = network.getConnectedEdges(nodeId) as number[]
        
        // Create sets for quick lookup
        const highlightedNodes = new Set([nodeId, ...connectedNodes])
        const highlightedEdges = new Set(connectedEdges)

        // Update all nodes
        const allNodes = nodes.get()
        const updatedNodes = allNodes.map(node => ({
          ...node,
          color: highlightedNodes.has(node.id) 
            ? {
                background: node.originalColor.background,
                border: node.originalColor.border,
                highlight: node.color.highlight,
                hover: node.color.hover
              }
            : {
                background: node.fadedColor.background,
                border: node.fadedColor.border,
                highlight: node.color.highlight,
                hover: node.color.hover
              }
        }))
        nodes.update(updatedNodes)

        // Update all edges
        const allEdges = edges.get()
        const updatedEdges = allEdges.map(edge => ({
          ...edge,
          color: highlightedEdges.has(edge.id) 
            ? {
                color: edge.originalColor,
                highlight: edge.color.highlight
              }
            : {
                color: edge.fadedColor,
                highlight: edge.color.highlight
              }
        }))
        edges.update(updatedEdges)
      } else {
        // Reset all nodes and edges to original colors
        const allNodes = nodes.get()
        const resetNodes = allNodes.map(node => ({
          ...node,
          color: {
            background: node.originalColor.background,
            border: node.originalColor.border,
            highlight: node.color.highlight,
            hover: node.color.hover
          }
        }))
        nodes.update(resetNodes)

        const allEdges = edges.get()
        const resetEdges = allEdges.map(edge => ({
          ...edge,
          color: {
            color: edge.originalColor,
            highlight: edge.color.highlight
          }
        }))
        edges.update(resetEdges)
      }
    }

    // Enhanced interactions
    network.on('click', (event) => {
      if (event.nodes.length > 0) {
        const nodeId = event.nodes[0]
        const node = data.nodes.find(n => n.id === nodeId)
        if (node && node.slug !== currentSlug) {
          window.location.href = `/${encodeURIComponent(node.slug)}`
        }
      }
    })

    // Hover events for connection highlighting
    network.on('hoverNode', (event) => {
      if (isGlobal) { // Only apply this effect to global graph
        highlightConnections(event.node)
      }
    })

    network.on('blurNode', () => {
      if (isGlobal) { // Only apply this effect to global graph
        highlightConnections(null)
      }
    })

    // Handle edge hover to show connected nodes
    network.on('hoverEdge', (event) => {
      if (isGlobal) {
        const edge = edges.get(event.edge)
        if (edge) {
          // Highlight both nodes connected by this edge
          highlightConnections(edge.from)
        }
      }
    })

    network.on('blurEdge', () => {
      if (isGlobal) {
        highlightConnections(null)
      }
    })

    // Track if this is the initial stabilization to prevent interfering with user interactions
    let isInitialStabilization = true

    // Better initial positioning and zoom handling
    network.on('stabilized', () => {
      // Only auto-fit on the very first stabilization, not on subsequent ones
      if (isInitialStabilization) {
        if (isGlobal) {
          network.fit()
        } else {
          network.fit({ scale: 1.1 })
        }
        isInitialStabilization = false
      }
    })

    // Ensure canvas is interactive and handles events properly
    network.on('ready', () => {
      const canvas = container.querySelector('canvas')
      if (canvas && !isGlobal) {
        canvas.style.cursor = 'grab'
        canvas.addEventListener('mousedown', () => {
          canvas.style.cursor = 'grabbing'
        })
        canvas.addEventListener('mouseup', () => {
          canvas.style.cursor = 'grab'
        })
        canvas.addEventListener('mouseleave', () => {
          canvas.style.cursor = 'grab'
        })
      }
    })

    return network
  }

  // Local graph effect
  useEffect(() => {
    if (!containerRef.current) return

    if (networkRef.current) {
      networkRef.current.destroy()
    }

    networkRef.current = createNetwork(containerRef.current, localData)

    return () => {
      if (networkRef.current) {
        networkRef.current.destroy()
        networkRef.current = null
      }
    }
  }, [localData, currentSlug])

  // Global graph effect - create network when panel content is visible
  useEffect(() => {
    // Only create global network if visible, we have data and container
    if (!globalGraphMounted || !globalContainerRef.current || !globalData.nodes.length) return

    // Clean up existing network first
    if (globalNetworkRef.current) {
      globalNetworkRef.current.destroy()
      globalNetworkRef.current = null
    }

    // Small delay to ensure container is properly mounted and visible
    const timer = setTimeout(() => {
      if (globalContainerRef.current && globalData.nodes.length && globalGraphMounted) {
        try {
          globalNetworkRef.current = createNetwork(globalContainerRef.current, globalData, true)
        } catch (error) {
          console.error('Failed to create global network:', error)
        }
      }
    }, 300)

    return () => {
      clearTimeout(timer)
      if (globalNetworkRef.current) {
        globalNetworkRef.current.destroy()
        globalNetworkRef.current = null
      }
    }
  }, [globalGraphMounted, globalData.nodes.length, globalData.edges.length, currentSlug]) // Include visibility state

  // Local panel graph effect
  useEffect(() => {
    if (!localPanelMounted || !localPanelContainerRef.current || !localData.nodes.length) return
    if (localPanelNetworkRef.current) {
      localPanelNetworkRef.current.destroy()
      localPanelNetworkRef.current = null
    }
    const timer = setTimeout(() => {
      if (localPanelContainerRef.current && localData.nodes.length && localPanelMounted) {
        try {
          localPanelNetworkRef.current = createNetwork(localPanelContainerRef.current, localData, false)
        } catch (error) {
          console.error('Failed to create local panel network:', error)
        }
      }
    }, 300)
    return () => {
      clearTimeout(timer)
      if (localPanelNetworkRef.current) {
        localPanelNetworkRef.current.destroy()
        localPanelNetworkRef.current = null
      }
    }
  }, [localPanelMounted, localData.nodes.length, localData.edges.length, currentSlug])

  // Tracker components
  const GlobalGraphTracker = () => {
    useEffect(() => {
      setGlobalGraphMounted(true)
      return () => setGlobalGraphMounted(false)
    }, [])
    return null
  }
  const LocalPanelTracker = () => {
    useEffect(() => {
      setLocalPanelMounted(true)
      return () => setLocalPanelMounted(false)
    }, [])
    return null
  }

  // Don't show local graph if there are no connections
  if (localData.nodes.length <= 1) {
    return null
  }

  return (
    <>
      <Card className={`${className} interactive-graph-card transition-all duration-200 hover:shadow-lg`}>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-3">
          <CardTitle className="text-sm font-medium text-foreground">Interactive Graph</CardTitle>
          <div className="flex gap-2">
            {/* Expand Local Graph Button */}
            <FloatingPanelRoot>
              <FloatingPanelTrigger
                title="Expand Local Graph"
                className="flex items-center gap-1 rounded-md border border-primary bg-primary text-primary-foreground px-3 py-1.5 text-xs font-semibold shadow hover:bg-primary/90 focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none transition-colors duration-150"
              >
                <Maximize2 className="h-3 w-3" />
                Expand
              </FloatingPanelTrigger>
              <FloatingPanelContent className="!fixed !inset-6 !left-1/2 !top-1/2 !-translate-x-1/2 !-translate-y-1/2 !w-[calc(100vw-3rem)] !max-w-5xl !h-[calc(100vh-3rem)] !max-h-[700px] !z-[100] !transform-gpu !overflow-hidden !flex !flex-col !h-full">
                <LocalPanelTracker />
                <FloatingPanelHeader className="p-4 border-b bg-card">
                  <div className="flex items-center justify-between">
                    <H2 className="text-xl font-bold border-none pb-0">Local Graph</H2>
                    <FloatingPanelCloseButton />
                  </div>
                </FloatingPanelHeader>
                <FloatingPanelBody className="flex-1 p-0 overflow-hidden !flex-1">
                  <div
                    ref={localPanelContainerRef}
                    className="w-full h-full flex-1 graph-container"
                    style={{ minHeight: '400px' }}
                  />
                </FloatingPanelBody>
                <FloatingPanelFooter className="p-4 border-t bg-card">
                  <div className="flex items-center justify-center text-xs w-full">
                    <Muted>{localData.nodes.length} nodes • {localData.edges.length} connections</Muted>
                  </div>
                </FloatingPanelFooter>
              </FloatingPanelContent>
            </FloatingPanelRoot>
            {/* Global Graph Button */}
            <FloatingPanelRoot>
              <FloatingPanelTrigger
                title="Global Graph"
                className="flex items-center gap-1 rounded-md border border-muted-foreground bg-background text-foreground px-3 py-1.5 text-xs font-semibold shadow hover:bg-accent focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none transition-colors duration-150"
          >
                <Globe className="h-3 w-3" />
            Global Graph
              </FloatingPanelTrigger>
              <FloatingPanelContent className="!fixed !inset-6 !left-1/2 !top-1/2 !-translate-x-1/2 !-translate-y-1/2 !w-[calc(100vw-3rem)] !max-w-7xl !h-[calc(100vh-3rem)] !max-h-[900px] !z-[100] !transform-gpu !overflow-hidden !flex !flex-col !h-full">
                <GlobalGraphTracker />
                <FloatingPanelHeader className="p-4 border-b bg-card">
                  <div className="flex items-center justify-between">
                    <div>
                      <H2 className="text-xl font-bold border-none pb-0">Interactive Graph</H2>
                      <Muted className="mt-1">
                        {globalData.nodes.length} pages • {globalData.edges.length} connections
                      </Muted>
                    </div>
                    <FloatingPanelCloseButton />
                  </div>
                </FloatingPanelHeader>
                <FloatingPanelBody className="flex-1 p-0 overflow-hidden !flex-1">
                  <div
                    ref={globalContainerRef}
                    className="w-full h-full flex-1 graph-container"
                    style={{ minHeight: '500px' }}
                  />
                </FloatingPanelBody>
                <FloatingPanelFooter className="p-4 border-t bg-card">
                  <div className="flex items-center justify-center space-x-8 text-sm w-full">
                    <div className="flex items-center">
                      <div className="w-4 h-4 rounded-full mr-2" style={{ backgroundColor: 'hsl(221 83% 53%)' }}></div>
                      <Muted>Current page</Muted>
                    </div>
                    <div className="flex items-center">
                      <div className="w-4 h-4 rounded-full mr-2" style={{ backgroundColor: 'hsl(210 50% 65%)' }}></div>
                      <Muted>Projects</Muted>
                    </div>
                    <div className="flex items-center">
                      <div className="w-4 h-4 rounded-full mr-2" style={{ backgroundColor: 'hsl(210 45% 75%)' }}></div>
                      <Muted>Other pages</Muted>
                    </div>
                  </div>
                </FloatingPanelFooter>
              </FloatingPanelContent>
            </FloatingPanelRoot>
          </div>
        </CardHeader>
        <CardContent className="pb-3">
          <div
            ref={containerRef}
            className="w-full h-64 rounded-md graph-container transition-all duration-200"
            style={{ minHeight: '256px', cursor: 'grab' }}
            onMouseDown={(e) => e.currentTarget.style.cursor = 'grabbing'}
            onMouseUp={(e) => e.currentTarget.style.cursor = 'grab'}
            onMouseLeave={(e) => e.currentTarget.style.cursor = 'grab'}
          />
          <Muted className="text-xs mt-2 text-center">
            {localData.nodes.length} nodes • {localData.edges.length} connections
          </Muted>
        </CardContent>
      </Card>
    </>
  )
} 