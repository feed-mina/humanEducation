'use client';
import Image from "next/image";

interface Props {
  id: string;
  meta: any;
  data: any;
}

export default function CardImage({ id, meta, data }: Props) {
  const src = data?.imageUrl || meta?.imageUrl || "/images/placeholder.png";
  const alt = data?.name || meta?.labelText || "";
  const mode = meta?.cssClass?.includes("circle") ? "circle" : "square";

  return (
    <div
      className={`card-image-wrapper ${mode === "circle" ? "rounded-full overflow-hidden" : "rounded-lg overflow-hidden"} relative w-full aspect-square`}
    >
      <Image src={src} alt={alt} fill className="object-cover" sizes="150px" />
    </div>
  );
}
