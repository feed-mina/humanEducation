'use client';
import { usePathname } from "next/navigation";

const HIDE_PATHS = ["/browse", "/movies", "/latest", "/intro4", "/intro5"];

export default function ConditionalHeader() {
  const pathname = usePathname();
  if (HIDE_PATHS.includes(pathname)) return null;
  return (
    <header className="flex items-center px-6 py-4 bg-black border-b border-gray-800">
      <span className="text-white font-bold text-xl tracking-tight">K-Ride</span>
    </header>
  );
}
