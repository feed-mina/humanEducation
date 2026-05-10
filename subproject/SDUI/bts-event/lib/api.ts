export async function guestChat(
  message: string,
  lang: "ko" | "en" | "ja",
  sessionId: string
): Promise<string> {
  const res = await fetch("/api/ai/guest/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, lang, sessionId }),
  });
  if (!res.ok) throw new Error("Chat request failed");
  const data = await res.json();
  const reply = data.reply || data.data?.reply;
  return reply as string;
}

export async function fetchBoardPosts(params: { pageSize: number; offset: number; filterId?: string }) {
  const sanitizedParams = Object.fromEntries(
    Object.entries(params).filter(([_, v]) => v !== undefined)
  );
  const query = new URLSearchParams(sanitizedParams as any).toString();
  const res = await fetch(`/api/execute/GET_FANBOARD_LIST?${query}`);
  return await res.json();
}

export async function submitBoardPost(postData: any) {
  const res = await fetch("/api/execute/INSERT_FANBOARD", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(postData),
  });
  return await res.json();
}

export async function updateBoardPost(postData: any) {
  const res = await fetch("/api/execute/UPDATE_FANBOARD", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(postData),
  });
  return await res.json();
}

export async function uploadFile(file: File) {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch("/api/ai/interview/resume/upload", {
    method: "POST",
    body: formData,
  });
  return await res.json();
}
