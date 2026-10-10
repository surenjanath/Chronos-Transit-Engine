// The app's own typeface (static/css/main.css: --sans "Instrument Sans") for both headings and body.
import { loadFont } from "@remotion/google-fonts/InstrumentSans";

const sans = loadFont("normal", { weights: ["400", "500", "600", "700"], subsets: ["latin"] });

export const OUTFIT = sans.fontFamily;
export const MANROPE = sans.fontFamily;
