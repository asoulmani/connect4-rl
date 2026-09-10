import { motion } from "framer-motion";

const ROWS = 6;
const COLS = 7;
const CELL = 72;

function discClass(value: number, winning: boolean, dim: boolean): string {
  const base = "absolute inset-[10%] rounded-full";
  const opacity = dim && !winning ? "opacity-35" : "opacity-100";
  if (value === 1) {
    return `${base} ${opacity} bg-gradient-to-br from-[#ff8aa0] to-[#ff335c] shadow-[inset_-6px_-8px_16px_rgba(0,0,0,0.35),0_0_18px_rgba(255,77,109,0.45)]`;
  }
  return `${base} ${opacity} bg-gradient-to-br from-[#ffe7a3] to-[#f5b942] shadow-[inset_-6px_-8px_16px_rgba(0,0,0,0.28),0_0_18px_rgba(255,209,102,0.4)]`;
}

export function Board({
  board,
  valid,
  winning,
  hoverCol,
  lastAction,
  interactive,
  onHover,
  onDrop,
}: {
  board: number[][];
  valid: number[];
  winning: number[][] | null;
  hoverCol: number | null;
  lastAction: number | null;
  interactive: boolean;
  onHover: (col: number | null) => void;
  onDrop: (col: number) => void;
}) {
  const win = new Set((winning ?? []).map(([r, c]) => `${r}-${c}`));
  const dim = Boolean(winning?.length);

  return (
    <div className="w-full max-w-[min(92vw,42rem)]">
      <div className="relative rounded-[28px] border border-white/10 bg-gradient-to-b from-[#2a3b5c] to-[#152033] p-3 shadow-board sm:p-4">
        <div
          className="grid"
          style={{
            gridTemplateColumns: `repeat(${COLS}, minmax(${CELL}px, 1fr))`,
            gridTemplateRows: `repeat(${ROWS}, minmax(${CELL}px, 1fr))`,
          }}
        >
          {Array.from({ length: ROWS * COLS }, (_, i) => {
            const row = Math.floor(i / COLS);
            const col = i % COLS;
            const value = board[row]?.[col] ?? 0;
            const isWin = win.has(`${row}-${col}`);
            const highlight = hoverCol === col && interactive && valid.includes(col);
            const isLatestInCol =
              lastAction === col &&
              value !== 0 &&
              (row === 0 || board[row - 1][col] === 0);

            return (
              <button
                key={`${row}-${col}`}
                type="button"
                disabled={!interactive || !valid.includes(col)}
                aria-label={`Drop in column ${col}`}
                onMouseEnter={() => onHover(col)}
                onMouseLeave={() => onHover(null)}
                onClick={() => onDrop(col)}
                className={`relative overflow-hidden rounded-full bg-[#070b14] m-1 ring-1 ring-black/60 disabled:cursor-default ${
                  highlight ? "ring-2 ring-cyan-300" : ""
                }`}
              >
                {value !== 0 ? (
                  <motion.div
                    className={discClass(value, isWin, dim)}
                    initial={isLatestInCol ? { y: -((row + 1) * CELL) } : { y: 0 }}
                    animate={{
                      y: 0,
                      scale: isWin ? [1, 1.07, 1] : 1,
                    }}
                    transition={{
                      y: { type: "spring", stiffness: 340, damping: 20 },
                      scale: isWin ? { repeat: Infinity, duration: 1.15 } : { duration: 0 },
                    }}
                  />
                ) : highlight ? (
                  <span className="absolute inset-[18%] rounded-full border-2 border-dashed border-cyan-200/50" />
                ) : null}
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}
