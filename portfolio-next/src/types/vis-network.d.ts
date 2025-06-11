declare module 'vis-network/standalone' {
  export interface Node {
    id: string
    label?: string
    title?: string
    color?: any
    font?: any
    shape?: string
    size?: number
    originalColor?: any
    fadedColor?: any
  }

  export interface Edge {
    id: number
    from: string
    to: string
    color?: any
    width?: number
    smooth?: any
    originalColor?: string
    fadedColor?: string
  }

  export interface NetworkOptions {
    nodes?: any
    edges?: any
    physics?: any
    interaction?: any
    layout?: any
  }

  export interface ClickEvent {
    nodes: string[]
    edges: number[]
  }

  export interface HoverEvent {
    node: string
  }

  export interface EdgeHoverEvent {
    edge: number
  }

  export interface ZoomEvent {
    scale: number
  }

  export class Network {
    constructor(container: HTMLElement, data: any, options?: NetworkOptions)
    on(event: string, callback: (params: any) => void): void
    on(event: 'click', callback: (params: ClickEvent) => void): void
    on(event: 'hoverNode', callback: (params: HoverEvent) => void): void
    on(event: 'blurNode', callback: () => void): void
    on(event: 'hoverEdge', callback: (params: EdgeHoverEvent) => void): void
    on(event: 'blurEdge', callback: () => void): void
    on(event: 'zoom', callback: (params: ZoomEvent) => void): void
    getConnectedNodes(nodeId: string): string[]
    getConnectedEdges(nodeId: string): number[]
    getScale(): number
    fit(options?: { scale?: number }): void
    destroy(): void
  }
}

declare module 'vis-data' {
  export class DataSet<T = any> {
    constructor(data?: T[])
    get(): T[]
    get(id: any): T
    update(data: T[]): void
  }
} 