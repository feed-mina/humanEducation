"use client";

import { useEffect } from "react";
import { X, Heart } from "lucide-react";

interface Props {
  onClose: () => void;
  lang: string;
}

export default function CheerMode({ onClose, lang }: Props) {
  useEffect(() => {
    // Keep the screen awake if possible? (Not reliable without Web API)
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = "auto";
    };
  }, []);

  return (
    <div className="cheer-overlay">
      <button onClick={onClose} className="cheer-close">
        <X size={24} />
      </button>
      
      <div className="cheer-content flex flex-col items-center">
        <Heart 
          size={120} 
          fill="white" 
          className="text-white mb-8 animate-ping opacity-75"
        />
        <h2 className="text-4xl font-black text-white mb-4 tracking-widest uppercase">
          {lang === "ko" ? "보라해 💜" : "BORAHAE 💜"}
        </h2>
        <p className="text-white/70 font-bold mb-10 text-lg uppercase tracking-wider">
          {lang === "ko" ? "광화문을 보라빛으로!" : "Purple Power!"}
        </p>
        
        <div className="grid grid-cols-4 gap-4 opacity-50">
          {[...Array(8)].map((_, i) => (
            <div key={i} className="w-4 h-4 rounded-full bg-white animate-pulse" style={{ animationDelay: `${i * 0.2}s` }} />
          ))}
        </div>
      </div>
      
      {/* Dynamic Background Noise */}
      <div className="absolute inset-0 pointer-events-none opacity-20 pointer-fine:mix-blend-overlay">
        <div className="w-full h-full bg-[radial-gradient(circle_at_center,rgba(255,255,255,0.2)_0%,transparent_70%)] animate-pulse" />
      </div>
    </div>
  );
}
