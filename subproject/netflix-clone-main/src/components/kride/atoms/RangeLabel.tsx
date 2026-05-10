'use client';

interface Props {
  id: string;
  meta: any;
  data: any;
  value?: number;
}
// 한국 비용이라는걸 표시
export default function RangeLabel({ value = 0 }: Props) {
  const formatted = `₩${value.toLocaleString("ko-KR")}`;
  return <span className="range-label text-white text-sm font-medium">{formatted}</span>;
}
