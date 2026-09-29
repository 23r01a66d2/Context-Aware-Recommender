import React from 'react';

export default function ModalityWeightBar({ weights, isColdStart = false }) {
  if (!weights) return null;

  const beh = Math.max(0, weights.behavior ?? 0);
  const cont = Math.max(0, weights.content ?? 0);
  const ctx = Math.max(0, weights.context ?? 0);
  const total = (beh + cont + ctx) || 1.0;

  const behPct = ((beh / total) * 100).toFixed(1);
  const contPct = ((cont / total) * 100).toFixed(1);
  const ctxPct = ((ctx / total) * 100).toFixed(1);

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-4 shadow-sm">
      <div className="flex items-center justify-between mb-2">
        <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
          Modality Attention Fusion Weights
          <span className="text-[10px] text-slate-500 font-normal ml-1.5 lowercase">
            (learned representational weights)
          </span>
        </h4>
        {isColdStart && (
          <span className="text-[11px] font-medium px-2 py-0.5 rounded-full bg-amber-500/10 text-amber-400 border border-amber-500/20 animate-pulse">
            Cold-Start Adaptation Active (Behavior Penalized)
          </span>
        )}
      </div>

      {/* Stacked Bar */}
      <div className="h-3 w-full bg-slate-800 rounded-full overflow-hidden flex shadow-inner">
        <div
          style={{ width: `${behPct}%` }}
          className={`h-full transition-all duration-500 ${
            beh <= 0.015 ? 'bg-slate-600 opacity-50' : 'bg-indigo-500'
          }`}
          title={`Behavior Weight: ${beh.toFixed(3)} (${behPct}%)`}
        />
        <div
          style={{ width: `${contPct}%` }}
          className="h-full bg-emerald-500 transition-all duration-500"
          title={`Content Weight: ${cont.toFixed(3)} (${contPct}%)`}
        />
        <div
          style={{ width: `${ctxPct}%` }}
          className="h-full bg-sky-500 transition-all duration-500"
          title={`Context Weight: ${ctx.toFixed(3)} (${ctxPct}%)`}
        />
      </div>

      {/* Legend & Details */}
      <div className="grid grid-cols-3 gap-2 mt-3 text-xs">
        <div className="flex items-center gap-2">
          <span className={`w-2.5 h-2.5 rounded-sm ${beh <= 0.015 ? 'bg-slate-500' : 'bg-indigo-500'}`}></span>
          <div>
            <div className="text-slate-400 font-medium">Behavioral</div>
            <div className="font-mono text-slate-200">
              {beh.toFixed(3)} <span className="text-slate-500 text-[10px]">({behPct}%)</span>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <span className="w-2.5 h-2.5 rounded-sm bg-emerald-500"></span>
          <div>
            <div className="text-slate-400 font-medium">Content / Meta</div>
            <div className="font-mono text-slate-200">
              {cont.toFixed(3)} <span className="text-slate-500 text-[10px]">({contPct}%)</span>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <span className="w-2.5 h-2.5 rounded-sm bg-sky-500"></span>
          <div>
            <div className="text-slate-400 font-medium">Contextual</div>
            <div className="font-mono text-slate-200">
              {ctx.toFixed(3)} <span className="text-slate-500 text-[10px]">({ctxPct}%)</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
