interface CatLiLogoProps {
  size?: number;
}

export default function CatLiLogo({ size = 48 }: CatLiLogoProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 64 64"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      {/* ── Bass drum ───────────────────────────────────────────────────── */}
      <rect x="11" y="44" width="36" height="16" rx="5" fill="#FFD0A5" stroke="#E8A86A" strokeWidth="1.5" />
      {/* drum head top */}
      <ellipse cx="29" cy="44" rx="18" ry="4.5" fill="#FFF3E0" stroke="#E8A86A" strokeWidth="1.5" />
      {/* drum tension rods */}
      <line x1="15" y1="44" x2="15" y2="60" stroke="#E8A86A" strokeWidth="0.8" />
      <line x1="22" y1="44" x2="22" y2="60" stroke="#E8A86A" strokeWidth="0.8" />
      <line x1="36" y1="44" x2="36" y2="60" stroke="#E8A86A" strokeWidth="0.8" />
      <line x1="43" y1="44" x2="43" y2="60" stroke="#E8A86A" strokeWidth="0.8" />

      {/* ── Snare drum (right) ──────────────────────────────────────────── */}
      <rect x="48" y="39" width="13" height="7" rx="2" fill="#E0D7FF" stroke="#9D8FDB" strokeWidth="1.2" />
      <ellipse cx="54.5" cy="39" rx="6.5" ry="2.5" fill="#F0ECFF" stroke="#9D8FDB" strokeWidth="1.2" />

      {/* ── Hi-hat (left) ───────────────────────────────────────────────── */}
      <ellipse cx="8" cy="37" rx="7" ry="2" fill="#E0D7FF" stroke="#9D8FDB" strokeWidth="1.2" />
      <ellipse cx="8" cy="34" rx="7" ry="2" fill="#F0ECFF" stroke="#9D8FDB" strokeWidth="1.2" />
      <line x1="8" y1="36" x2="8" y2="48" stroke="#C4B5FD" strokeWidth="1.2" />

      {/* ── Cat body ────────────────────────────────────────────────────── */}
      <ellipse cx="29" cy="37" rx="7" ry="8" fill="#DDD5FF" stroke="#9D8FDB" strokeWidth="1.5" />

      {/* ── Cat head ────────────────────────────────────────────────────── */}
      <circle cx="29" cy="20" r="12" fill="#EDE8FF" stroke="#9D8FDB" strokeWidth="1.5" />

      {/* ── Ears ────────────────────────────────────────────────────────── */}
      {/* left ear */}
      <polygon points="19,13 16,4 25,11" fill="#EDE8FF" stroke="#9D8FDB" strokeWidth="1.5" strokeLinejoin="round" />
      <polygon points="19,12 17,6 24,11" fill="#FFD0A5" />
      {/* right ear */}
      <polygon points="39,13 42,4 33,11" fill="#EDE8FF" stroke="#9D8FDB" strokeWidth="1.5" strokeLinejoin="round" />
      <polygon points="39,12 41,6 34,11" fill="#FFD0A5" />

      {/* ── Face ────────────────────────────────────────────────────────── */}
      {/* eyes */}
      <circle cx="24" cy="19" r="2.2" fill="#2D2B4E" />
      <circle cx="34" cy="19" r="2.2" fill="#2D2B4E" />
      {/* eye shine */}
      <circle cx="25" cy="18" r="0.9" fill="white" />
      <circle cx="35" cy="18" r="0.9" fill="white" />
      {/* nose */}
      <ellipse cx="29" cy="23" rx="1.5" ry="1" fill="#FFB0C8" />
      {/* mouth */}
      <path d="M27 24.5 Q29 26.5 31 24.5" stroke="#9D8FDB" strokeWidth="1" strokeLinecap="round" fill="none" />
      {/* whiskers */}
      <line x1="18" y1="23" x2="27" y2="24" stroke="#C4B5FD" strokeWidth="0.8" strokeLinecap="round" />
      <line x1="18" y1="25.5" x2="27" y2="25.5" stroke="#C4B5FD" strokeWidth="0.8" strokeLinecap="round" />
      <line x1="40" y1="23" x2="31" y2="24" stroke="#C4B5FD" strokeWidth="0.8" strokeLinecap="round" />
      <line x1="40" y1="25.5" x2="31" y2="25.5" stroke="#C4B5FD" strokeWidth="0.8" strokeLinecap="round" />

      {/* ── Drumsticks / arms ───────────────────────────────────────────── */}
      {/* left arm → hi-hat */}
      <line x1="23" y1="33" x2="10" y2="30" stroke="#9D8FDB" strokeWidth="2" strokeLinecap="round" />
      <line x1="10" y1="30" x2="4" y2="26" stroke="#C4B5FD" strokeWidth="1.5" strokeLinecap="round" />
      {/* right arm → snare */}
      <line x1="35" y1="33" x2="47" y2="35" stroke="#9D8FDB" strokeWidth="2" strokeLinecap="round" />
      <line x1="47" y1="35" x2="57" y2="30" stroke="#C4B5FD" strokeWidth="1.5" strokeLinecap="round" />

      {/* ── Tail (peeking left) ─────────────────────────────────────────── */}
      <path d="M22 42 Q10 50 14 58" stroke="#C4B5FD" strokeWidth="2.5" strokeLinecap="round" fill="none" />
    </svg>
  );
}
