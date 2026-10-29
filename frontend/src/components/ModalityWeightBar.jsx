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
    <div className="dark:bg-navy-900 bg-white border dark:border-navy-800 border-slate-200 rounded-2xl p-4 shadow-sm transition-colors duration-200">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-3">
        <div>
          <h4 className="text-xs font-semibold uppercase tracking-wider dark:text-muted text-slate-500">
            Adaptive Representation Fusion Contribution
          </h4>
          <p className="text-sm font-bold dark:text-coolwhite text-slate-900 mt-0.5">
            <span className="text-[#3B82F6] dark:text-[#60A5FA]">Behavior: {behPct}%</span>
            <span className="dark:text-navy-700 text-slate-300 mx-1.5">|</span>
            <span className="text-teal-600 dark:text-teal-400">Content: {contPct}%</span>
            <span className="dark:text-navy-700 text-slate-300 mx-1.5">|</span>
            <span className="text-sky-600 dark:text-[#38BDF8]">Context: {ctxPct}%</span>
          </p>
        </div>
        {isColdStart && (
          <span className="text-[11px] font-medium px-2.5 py-1 rounded-full bg-amber-500/10 text-amber-700 dark:text-amber-400 border border-amber-300 dark:border-amber-500/20 flex items-center gap-1.5 self-start sm:self-auto">
            <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-ping"></span>
            Cold-Start Active: Behavior Gated, Content & Context Emphasized
          </span>
        )}
      </div>

      {/* Stacked Bar */}
      <div className="h-2.5 w-full dark:bg-navy-950 bg-slate-100 rounded-full overflow-hidden flex shadow-inner mb-3">
        <div
          style={{ width: `${behPct}%` }}
          className={`h-full transition-all duration-500 ${
            beh <= 0.02 ? 'dark:bg-navy-700 bg-slate-300 opacity-60' : 'bg-[#3B82F6] dark:bg-[#60A5FA]'
          }`}
          title={`Behavior Signal: ${beh.toFixed(3)} (${behPct}%)`}
        />
        <div
          style={{ width: `${contPct}%` }}
          className="h-full bg-teal-500 transition-all duration-500"
          title={`Content (Image + Text / Metadata): ${cont.toFixed(3)} (${contPct}%)`}
        />
        <div
          style={{ width: `${ctxPct}%` }}
          className="h-full bg-sky-500 transition-all duration-500"
          title={`Session Context: ${ctx.toFixed(3)} (${ctxPct}%)`}
        />
      </div>

      {/* 3 Modality Breakdown Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5 text-xs">
        {/* Behavior Signal Card */}
        <div className={`p-2.5 rounded-xl border ${beh <= 0.02 ? 'dark:bg-navy-950/60 bg-slate-50 dark:border-navy-800 border-slate-200 dark:text-muted text-slate-400' : 'dark:bg-[#60A5FA]/10 bg-blue-50/70 border-blue-200 dark:border-[#60A5FA]/30 text-blue-900 dark:text-slate-200'}`}>
          <div className="flex items-center gap-1.5 font-medium mb-1">
            <span className={`w-2 h-2 rounded-full ${beh <= 0.02 ? 'bg-slate-400' : 'bg-[#3B82F6] dark:bg-[#60A5FA]'}`}></span>
            <span className="dark:text-coolwhite text-slate-800 font-semibold">Behavior Signal</span>
          </div>
          <div className="text-base font-bold font-mono dark:text-coolwhite text-slate-900">
            {behPct}% <span className="text-xs font-normal dark:text-muted text-slate-500">({beh.toFixed(3)})</span>
          </div>
          <div className="text-[11px] dark:text-muted text-slate-500 mt-0.5">
            {isColdStart ? 'Gated down for zero-history user' : 'Recent interactions & latent affinity'}
          </div>
        </div>

        {/* Content Card */}
        <div className="p-2.5 rounded-xl border dark:bg-teal-500/10 bg-teal-50/70 border-teal-200 dark:border-teal-500/30 text-teal-900 dark:text-slate-200">
          <div className="flex items-center gap-1.5 font-medium mb-1">
            <span className="w-2 h-2 rounded-full bg-teal-500"></span>
            <span className="dark:text-coolwhite text-slate-800 font-semibold">Content (Image + Text / Meta)</span>
          </div>
          <div className="text-base font-bold font-mono dark:text-coolwhite text-slate-900">
            {contPct}% <span className="text-xs font-normal dark:text-muted text-slate-500">({cont.toFixed(3)})</span>
          </div>
          <div className="text-[11px] dark:text-muted text-slate-500 mt-0.5">
            Embeddings, categories & item attributes
          </div>
        </div>

        {/* Session Context Card */}
        <div className="p-2.5 rounded-xl border dark:bg-[#38BDF8]/10 bg-sky-50/70 border-sky-200 dark:border-[#38BDF8]/30 text-sky-900 dark:text-slate-200">
          <div className="flex items-center gap-1.5 font-medium mb-1">
            <span className="w-2 h-2 rounded-full bg-sky-500"></span>
            <span className="dark:text-coolwhite text-slate-800 font-semibold">Session Context</span>
          </div>
          <div className="text-base font-bold font-mono dark:text-coolwhite text-slate-900">
            {ctxPct}% <span className="text-xs font-normal dark:text-muted text-slate-500">({ctx.toFixed(3)})</span>
          </div>
          <div className="text-[11px] dark:text-muted text-slate-500 mt-0.5">
            Device, timing & real-time session inputs
          </div>
        </div>
      </div>
    </div>
  );
}
