import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata = {
  title: "Heart Health Predictor",
  keywords: [
    "Heart Health",
    "Cardiovascular Disease",
    "Health Prediction",
    "AI",
    "Machine Learning",
    "Next.js",
    "React",
  ],
  authors: [
    {
      name: "Robin,Siam",
      
  description: "heart health predictor using AI and machine learning",
    },
  ],
  creator: "Robin",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
