export const EXCERPTS = [
  { id: "singing", title: "Singing", detail: "Mostly voice." },
  { id: "krar", title: "Krar", detail: "Mostly krar." },
  { id: "both", title: "Both", detail: "Singing and krar together." },
] as const;

export const TRACKS = [
  { id: "original", label: "Original", file: "original.wav" },
  { id: "kim_inst_instrumental", label: "Kim instrumental", file: "kim_inst_instrumental.wav" },
  { id: "kim_inst_vocals", label: "Kim vocals", file: "kim_inst_vocals.wav" },
  { id: "vocft_instrumental", label: "Voc_FT instrumental", file: "vocft_instrumental.wav" },
  { id: "vocft_vocals", label: "Voc_FT vocals", file: "vocft_vocals.wav" },
] as const;

export const CHOICES = [
  { id: "kim_inst", label: "Kim Inst" },
  { id: "vocft", label: "Voc_FT" },
  { id: "same", label: "No clear difference" },
  { id: "unsure", label: "Not sure" },
] as const;

export type ExcerptId = (typeof EXCERPTS)[number]["id"];
export type TrackId = (typeof TRACKS)[number]["id"];
export type ChoiceId = (typeof CHOICES)[number]["id"];

export type ExcerptJudgment = {
  preservesKrar: ChoiceId | null;
  removesSinging: ChoiceId | null;
  note: string;
  updatedAt: string | null;
};

export type ListeningNotes = Record<ExcerptId, ExcerptJudgment>;

export function emptyNotes(): ListeningNotes {
  return {
    singing: { preservesKrar: null, removesSinging: null, note: "", updatedAt: null },
    krar: { preservesKrar: null, removesSinging: null, note: "", updatedAt: null },
    both: { preservesKrar: null, removesSinging: null, note: "", updatedAt: null },
  };
}

export function isExcerptId(value: string): value is ExcerptId {
  return EXCERPTS.some((excerpt) => excerpt.id === value);
}

export function isTrackId(value: string): value is TrackId {
  return TRACKS.some((track) => track.id === value);
}

export function isChoiceId(value: string): value is ChoiceId {
  return CHOICES.some((choice) => choice.id === value);
}
