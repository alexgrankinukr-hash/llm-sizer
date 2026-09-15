/** Share link, image download and the presentation link. */
import { useState } from 'react';
import { analytics } from '../../lib/llm-sizer/app/analytics';
import type { View } from '../../lib/llm-sizer/app/state';
import { DownloadIcon, LinkIcon } from './icons';

export interface ShareExportProps {
  encoded: string;
  shareUrl: string;
  view: View['kind'];
}

export function ShareExport({ encoded, shareUrl, view }: ShareExportProps) {
  const [toast, setToast] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  function say(text: string) {
    setToast(text);
    window.setTimeout(() => setToast(null), 2600);
  }
  async function share() {
    const url = shareUrl || window.location.href;
    if (navigator.share && /Mobi|Android|iPhone|iPad/.test(navigator.userAgent)) {
      try {
        await navigator.share({ title: 'LLM Sizer', url });
        analytics.shareLink('share');
        return;
      } catch {
        /* cancelled: fall through to copy */
      }
    }
    try {
      await navigator.clipboard.writeText(url);
      say('Link copied');
    } catch {
      say(url);
    }
    analytics.shareLink('copy');
  }
  async function download() {
    setBusy(true);
    try {
      const imageView = view === 'memory' || view === 'machine' ? 'table' : view; // the memory chart and the machine view have no image yet
      const res = await fetch(`/api/llm-sizer/image.png?s=${encodeURIComponent(encoded)}&size=table&view=${imageView}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const blob = await res.blob();
      const file = new File([blob], 'llm-sizer.png', { type: 'image/png' });
      if (navigator.canShare?.({ files: [file] }) && /iPhone|iPad/.test(navigator.userAgent)) {
        await navigator.share({ files: [file], title: 'LLM Sizer' });
      } else {
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = 'llm-sizer.png';
        a.click();
        window.setTimeout(() => URL.revokeObjectURL(a.href), 4000);
      }
      analytics.imageExported(imageView, 'table');
      say('Image ready');
    } catch {
      say('Image export is not available right now');
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <button type="button" className="lls-chip !py-1.5 !px-3" onClick={share}>
        <LinkIcon size={14} /> Share link
      </button>
      <button type="button" className="lls-chip !py-1.5 !px-3" onClick={download} disabled={busy}>
        <DownloadIcon size={14} /> {busy ? 'Rendering…' : 'Image'}
      </button>
      {toast && (
        <div role="status" aria-live="polite" className="fixed bottom-6 left-1/2 -translate-x-1/2 z-[90] bg-[#1A1A1A] text-white text-sm px-5 py-3 shadow-xl whitespace-nowrap max-w-[90vw] overflow-hidden text-ellipsis">
          {toast}
        </div>
      )}
    </>
  );
}
