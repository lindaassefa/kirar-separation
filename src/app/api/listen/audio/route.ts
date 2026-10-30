import { createReadStream } from "node:fs";
import { access } from "node:fs/promises";
import path from "node:path";
import { Readable } from "node:stream";

import type { NextRequest } from "next/server";

import { isExcerptId, isTrackId, TRACKS } from "@/lib/listen-catalog";

const ROOT = path.join(process.cwd(), "separator", "runs", "vocal-compare");

export async function GET(request: NextRequest) {
  const excerpt = request.nextUrl.searchParams.get("excerpt") ?? "";
  const track = request.nextUrl.searchParams.get("track") ?? "";
  const fileName = TRACKS.find((item) => item.id === track)?.file;
  if (!isExcerptId(excerpt) || !isTrackId(track) || !fileName) {
    return Response.json({ error: "That recording is missing." }, { status: 404 });
  }

  const file = path.resolve(ROOT, excerpt, fileName);
  const root = path.resolve(ROOT);
  if (file !== path.join(root, excerpt, fileName)) {
    return Response.json({ error: "That recording is missing." }, { status: 404 });
  }

  try {
    await access(file);
  } catch {
    return Response.json({ error: "That recording is missing." }, { status: 404 });
  }

  const stream = Readable.toWeb(createReadStream(file)) as ReadableStream;
  return new Response(stream, {
    headers: {
      "Content-Type": "audio/wav",
      "Cache-Control": "no-store",
    },
  });
}
