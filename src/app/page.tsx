import Separation from "@/components/separation";

export default function Home() {
  return (
    <>
      <header className="mx-auto flex w-full max-w-6xl items-center justify-between px-5 pt-6 sm:px-8">
        <p className="font-display text-sm font-semibold tracking-tight">Kirar Separation</p>
        <p className="text-xs tracking-wide text-lilac uppercase">Krar experimental</p>
      </header>

      <main className="mx-auto flex w-full max-w-6xl flex-1 flex-col gap-10 px-5 pt-10 pb-16 sm:px-8">
        <div className="flex max-w-2xl flex-col gap-3">
          <h1 className="font-display text-4xl leading-none font-semibold tracking-tight sm:text-6xl">
            Separate a recording.
          </h1>
          <p className="text-base leading-7 text-muted sm:text-lg">
            Upload a song to separate vocals, drums, bass, and other instruments. Krar isolation stays experimental and is not available.
          </p>
        </div>
        <Separation />
      </main>
    </>
  );
}
