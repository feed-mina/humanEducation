"use client";

import { useState } from "react";
import { X, ExternalLink, Play } from "lucide-react";

export default function LivePip() {
  const [isOpen, setIsOpen] = useState(true);
  const videoUrl = "https://www.youtube.com/embed/dRXKooC6tTM?autoplay=1&mute=1";

  if (!isOpen) return (
    <button 
      onClick={() => setIsOpen(true)}
      className="absolute bottom-20 right-20 z-50 w-12 h-12 bg-indigo-600 rounded-full flex items-center justify-center shadow-lg border-2 border-white text-white hover:scale-110 transition-all animate-bounce"
    >
      <Play size={20} fill="currentColor" />
    </button>
  );

  return (
    <div className="pip-window">
      <div className="flex items-center justify-between px-2 py-1 bg-black/80 text-[10px] text-white/50">
        <span className="flex items-center gap-1">
          <div className="w-1.5 h-1.5 rounded-full bg-red-600 animate-pulse" />
          LIVE
        </span>
        <div className="flex items-center gap-2">
          <a href="https://youtube.com/live/dRXKooC6tTM" target="_blank" rel="noopener">
            <ExternalLink size={10} className="hover:text-white" />
          </a>
          <button onClick={() => setIsOpen(false)}>
            <X size={10} className="hover:text-white" />
          </button>
        </div>
      </div>
      <iframe 
        src={videoUrl}
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
        allowFullScreen
      />
    </div>
  );
}
