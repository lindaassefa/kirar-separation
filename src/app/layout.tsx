import type { Metadata } from "next";
import { IBM_Plex_Sans, Syne } from "next/font/google";
import "./globals.css";

const plex = IBM_Plex_Sans({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  variable: "--font-plex",
  display: "swap",
});

const syne = Syne({
  subsets: ["latin"],
  variable: "--font-syne",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Kirar Separation",
  description: "Upload a song to separate vocals, drums, bass, and other instruments. Krar isolation is experimental and is not available.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${plex.variable} ${syne.variable} h-full antialiased`}>
      <body className="flex min-h-full flex-col bg-background text-foreground">{children}</body>
    </html>
  );
}
