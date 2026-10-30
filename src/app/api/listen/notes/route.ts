import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";

import type { NextRequest } from "next/server";

import {
  emptyNotes,
  isChoiceId,
  isExcerptId,
  type ChoiceId,
  type ExcerptJudgment,
  type ListeningNotes,
} from "@/lib/listen-catalog";

const NOTES = path.join(process.cwd(), "separator", "runs", "vocal-compare", "listening.json");

function choice(value: unknown): ChoiceId | null {
  if (value === null) return null;
  return typeof value === "string" && isChoiceId(value) ? value : null;
}

function judgment(value: unknown): ExcerptJudgment {
  const record = value && typeof value === "object" ? (value as Record<string, unknown>) : {};
  const note = typeof record.note === "string" ? record.note.slice(0, 2000) : "";
  const updatedAt = typeof record.updatedAt === "string" ? record.updatedAt : null;
  return {
    preservesKrar: choice(record.preservesKrar),
    removesSinging: choice(record.removesSinging),
    note,
    updatedAt,
  };
}

async function readNotes(): Promise<ListeningNotes> {
  const notes = emptyNotes();
  try {
    const parsed = JSON.parse(await readFile(NOTES, "utf8")) as unknown;
    if (!parsed || typeof parsed !== "object") return notes;
    for (const excerpt of Object.keys(notes) as (keyof ListeningNotes)[]) {
      notes[excerpt] = judgment((parsed as Record<string, unknown>)[excerpt]);
    }
  } catch {
    return notes;
  }
  return notes;
}

export async function GET() {
  return Response.json(await readNotes());
}

export async function POST(request: NextRequest) {
  const body = (await request.json()) as unknown;
  if (!body || typeof body !== "object") {
    return Response.json({ error: "The note could not be saved." }, { status: 400 });
  }
  const record = body as Record<string, unknown>;
  const excerpt = typeof record.excerpt === "string" ? record.excerpt : "";
  if (!isExcerptId(excerpt)) {
    return Response.json({ error: "The note could not be saved." }, { status: 400 });
  }
  if (record.preservesKrar !== null && !choice(record.preservesKrar) && record.preservesKrar !== undefined) {
    return Response.json({ error: "The note could not be saved." }, { status: 400 });
  }
  if (record.removesSinging !== null && !choice(record.removesSinging) && record.removesSinging !== undefined) {
    return Response.json({ error: "The note could not be saved." }, { status: 400 });
  }

  const notes = await readNotes();
  const current = notes[excerpt];
  notes[excerpt] = {
    preservesKrar: record.preservesKrar === undefined ? current.preservesKrar : choice(record.preservesKrar),
    removesSinging: record.removesSinging === undefined ? current.removesSinging : choice(record.removesSinging),
    note: typeof record.note === "string" ? record.note.slice(0, 2000) : current.note,
    updatedAt: new Date().toISOString(),
  };
  await writeFile(NOTES, `${JSON.stringify(notes, null, 2)}\n`);
  return Response.json(notes);
}
