/** Reorder a short list by dragging a grip (mouse and touch, via pointer events) or with the arrow keys on it. */
import { useRef, useState } from 'react';

interface Drag {
  from: number;
  over: number;
}

export function useListReorder(count: number, onMove: (from: number, to: number) => void, announce?: (text: string) => void) {
  const listRef = useRef<HTMLUListElement>(null);
  const dragRef = useRef<Drag | null>(null);
  const [drag, setDrag] = useState<Drag | null>(null);

  function indexAtY(y: number): number {
    const items = listRef.current ? Array.from(listRef.current.querySelectorAll<HTMLElement>('[data-reorder-item]')) : [];
    for (let i = 0; i < items.length; i++) {
      const r = items[i].getBoundingClientRect();
      if (y < r.top + r.height / 2) return i;
    }
    return Math.max(0, items.length - 1);
  }

  function handleProps(i: number, label: string) {
    return {
      'data-reorder-handle': '',
      'aria-label': `Reorder ${label}: drag, or press the arrow keys`,
      style: { touchAction: 'none' } as React.CSSProperties,
      onPointerDown: (e: React.PointerEvent<HTMLElement>) => {
        if (e.pointerType === 'mouse' && e.button !== 0) return;
        e.currentTarget.setPointerCapture(e.pointerId);
        dragRef.current = { from: i, over: i };
        setDrag(dragRef.current);
      },
      onPointerMove: (e: React.PointerEvent<HTMLElement>) => {
        const d = dragRef.current;
        if (!d) return;
        const over = indexAtY(e.clientY);
        if (over !== d.over) {
          dragRef.current = { from: d.from, over };
          setDrag(dragRef.current);
        }
      },
      onPointerUp: () => {
        const d = dragRef.current;
        dragRef.current = null;
        setDrag(null);
        if (d && d.over !== d.from) {
          onMove(d.from, d.over);
          announce?.(`${label} moved to position ${d.over + 1} of ${count}`);
        }
      },
      onPointerCancel: () => {
        dragRef.current = null;
        setDrag(null);
      },
      onKeyDown: (e: React.KeyboardEvent<HTMLElement>) => {
        const to = e.key === 'ArrowUp' ? i - 1 : e.key === 'ArrowDown' ? i + 1 : null;
        if (to === null || to < 0 || to >= count) return;
        e.preventDefault();
        onMove(i, to);
        announce?.(`${label} moved to position ${to + 1} of ${count}`);
        requestAnimationFrame(() => listRef.current?.querySelectorAll<HTMLElement>('[data-reorder-handle]')[to]?.focus());
      },
    };
  }

  function itemProps(i: number) {
    const cls = drag?.from === i ? 'lls-dragging' : drag && drag.over === i && drag.from !== i ? 'lls-drop' : '';
    return { 'data-reorder-item': '', className: cls };
  }

  return { listRef, drag, handleProps, itemProps };
}
