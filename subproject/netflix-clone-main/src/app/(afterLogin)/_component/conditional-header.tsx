'use client';
import { usePathname } from "next/navigation";
import Header from "./header";

const HIDE_PATHS = ["/browse", "/movies", "/latest", "/intro4", "/intro5"];

export default function ConditionalHeader() {
  const pathname = usePathname();
  if (HIDE_PATHS.includes(pathname)) return null;
  return <Header />;
}
