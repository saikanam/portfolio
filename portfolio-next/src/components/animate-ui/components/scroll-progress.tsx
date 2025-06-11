'use client';

import * as React from 'react';
import {
  motion,
  useScroll,
  useSpring,
  type HTMLMotionProps,
} from 'motion/react';

import { cn } from '@/lib/utils';

type ScrollProgressProps = React.ComponentProps<'div'> & {
  progressProps?: HTMLMotionProps<'div'>;
};

function ScrollProgress({
  ref,
  className,
  children,
  progressProps,
  ...props
}: ScrollProgressProps) {
  const containerRef = React.useRef<HTMLDivElement | null>(null);
  React.useImperativeHandle(ref, () => containerRef.current as HTMLDivElement);

  const { scrollYProgress } = useScroll(
    children ? { container: containerRef } : undefined,
  );

  const scaleX = useSpring(scrollYProgress, {
    stiffness: 250,
    damping: 40,
    bounce: 0,
  });

  return (
    <>
      <motion.div
        data-slot="scroll-progress"
        {...progressProps}
        style={{ 
          scaleX,
          position: undefined,
          ...progressProps?.style 
        }}
        className={cn(
          'scroll-progress',
          progressProps?.className,
        )}
      />
      {children && (
        <div
          ref={containerRef}
          data-slot="scroll-progress-container"
          className={cn('overflow-y-auto h-full', className)}
          {...props}
        >
          {children}
        </div>
      )}
    </>
  );
}

export { ScrollProgress, type ScrollProgressProps };
