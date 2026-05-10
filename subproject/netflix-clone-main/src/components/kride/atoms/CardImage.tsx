'use client';
import Image from "next/image";

interface Props {
  id: string;
  meta: any;
  data: any;
}
// DB의 css 칼럼으로 원형, 네모 지정 
export default function CardImage({ id, meta, data }: Props) {
  const src = data?.imageUrl || meta?.imageUrl || "/images/placeholder.png";
  const alt = data?.name || meta?.labelText || "";
  const mode = meta?.cssClass?.includes("circle") ? "circle" : "square";

  return (
    // [메모] mode에 사용한 문법에 대해 알려주세요
    <div
      className={`card-image-wrapper ${mode === "circle" ? "rounded-full overflow-hidden" : "rounded-lg overflow-hidden"} relative w-full aspect-square`}
    >
      <Image src={src} alt={alt} fill className="object-cover" sizes="150px" />
    </div>
  );
}
