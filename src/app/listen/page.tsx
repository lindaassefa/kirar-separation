import ListenComparison from "@/components/listen-comparison";

export const metadata = {
  title: "Listening comparison",
  description: "Compare Kim Inst and Voc_FT on three excerpts.",
};

export default function ListenPage() {
  return (
    <main className="mx-auto flex w-full max-w-6xl flex-1 flex-col gap-10 px-5 pt-10 pb-16 sm:px-8">
      <div className="flex max-w-2xl flex-col gap-3">
        <p className="text-xs tracking-wide text-lilac uppercase">Listening comparison</p>
        <h1 className="font-display text-4xl leading-none font-semibold tracking-tight sm:text-6xl">
          Compare Kim Inst and Voc_FT.
        </h1>
        <p className="text-base leading-7 text-muted sm:text-lg">
          One source plays at a time. Switching source keeps the same position. Instrumental is the vocal-removal
          estimate, not isolated krar. The singing excerpt is much quieter than the other two.
        </p>
      </div>
      <ListenComparison />
    </main>
  );
}
