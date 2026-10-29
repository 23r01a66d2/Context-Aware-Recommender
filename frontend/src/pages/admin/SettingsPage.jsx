import React, { useState } from 'react';
import { useTheme } from '../../context/ThemeContext';
import { useClient } from '../../context/ClientContext';
import ThemeSwitcher from '../../components/ThemeSwitcher';
import {
  Settings,
  Sun,
  Moon,
  Laptop,
  CheckCircle2,
  Database,
  Shield,
  Sliders,
  Cpu,
  RefreshCw,
  Bell,
  Save,
  Check
} from 'lucide-react';

export default function SettingsPage() {
  const { themeMode, resolvedTheme, setThemeMode } = useTheme();
  const { clients, selectedClientId, selectClient, health } = useClient();
  const [saveSuccess, setSaveSuccess] = useState(false);

  const [inferenceTimeout, setInferenceTimeout] = useState('5000');
  const [defaultTopK, setDefaultTopK] = useState('5');
  const [enableTelemetry, setEnableTelemetry] = useState(true);
  const [autoDwellTracking, setAutoDwellTracking] = useState(true);

  const handleSave = (e) => {
    e.preventDefault();
    setSaveSuccess(true);
    setTimeout(() => setSaveSuccess(false), 3000);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl sm:text-2xl font-bold dark:text-coolwhite text-slate-900 tracking-tight flex items-center gap-2">
            <Settings className="w-5 h-5 text-teal-500" />
            Platform & Theme Settings
          </h1>
          <p className="text-xs dark:text-muted text-slate-500 mt-0.5">
            Manage appearance themes, telemetry preferences, and global inference parameters.
          </p>
        </div>

        {saveSuccess && (
          <span className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-teal-500/15 text-teal-600 dark:text-teal-300 border border-teal-500/30 animate-fadeIn">
            <Check className="w-4 h-4 text-teal-500" /> Preferences Saved
          </span>
        )}
      </div>

      <form onSubmit={handleSave} className="space-y-6">
        {/* Appearance & Theme Section */}
        <div className="dark:bg-navy-900 bg-white border dark:border-navy-800 border-slate-200 rounded-2xl p-6 shadow-sm space-y-4">
          <div className="flex items-center justify-between border-b dark:border-navy-800 border-slate-200 pb-3">
            <div>
              <h2 className="text-sm font-bold dark:text-coolwhite text-slate-900 flex items-center gap-2">
                <Sun className="w-4 h-4 text-amber-500" />
                Theme & Display Appearance
              </h2>
              <p className="text-xs dark:text-muted text-slate-500 mt-0.5">
                Switch between Deep Navy Dark mode, Clean Light mode, or follow your OS system preferences.
              </p>
            </div>
            <ThemeSwitcher />
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-2">
            {/* Dark Card */}
            <div
              onClick={() => setThemeMode('dark')}
              className={`p-4 rounded-xl border cursor-pointer transition-all flex flex-col justify-between gap-3 ${
                themeMode === 'dark'
                  ? 'border-teal-500 bg-teal-500/5 ring-2 ring-teal-500/20'
                  : 'dark:bg-navy-950/60 bg-slate-50 dark:border-navy-800 border-slate-200 hover:border-slate-300 dark:hover:border-navy-700'
              }`}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Moon className="w-4 h-4 text-teal-400" />
                  <span className="text-xs font-bold dark:text-coolwhite text-slate-800">Dark Mode</span>
                </div>
                {themeMode === 'dark' && <CheckCircle2 className="w-4 h-4 text-teal-500" />}
              </div>
              <div className="h-14 rounded-lg bg-[#071426] border border-[#1E2E48] p-2 flex flex-col justify-between">
                <div className="flex gap-1.5">
                  <div className="w-3 h-3 rounded-full bg-teal-500"></div>
                  <div className="w-12 h-2 rounded bg-[#101F35]"></div>
                </div>
                <div className="w-full h-3 rounded bg-[#101F35]"></div>
              </div>
              <p className="text-[11px] dark:text-muted text-slate-500">
                Deep navy `#071426` surfaces with teal accents and high contrast typography.
              </p>
            </div>

            {/* Light Card */}
            <div
              onClick={() => setThemeMode('light')}
              className={`p-4 rounded-xl border cursor-pointer transition-all flex flex-col justify-between gap-3 ${
                themeMode === 'light'
                  ? 'border-teal-500 bg-teal-500/5 ring-2 ring-teal-500/20'
                  : 'dark:bg-navy-950/60 bg-slate-50 dark:border-navy-800 border-slate-200 hover:border-slate-300 dark:hover:border-navy-700'
              }`}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Sun className="w-4 h-4 text-amber-500" />
                  <span className="text-xs font-bold dark:text-coolwhite text-slate-800">Light Mode</span>
                </div>
                {themeMode === 'light' && <CheckCircle2 className="w-4 h-4 text-teal-500" />}
              </div>
              <div className="h-14 rounded-lg bg-[#F8FAFC] border border-[#CBD5E1] p-2 flex flex-col justify-between">
                <div className="flex gap-1.5">
                  <div className="w-3 h-3 rounded-full bg-teal-600"></div>
                  <div className="w-12 h-2 rounded bg-white"></div>
                </div>
                <div className="w-full h-3 rounded bg-white"></div>
              </div>
              <p className="text-[11px] dark:text-muted text-slate-500">
                Crisp `#F8FAFC` backgrounds, soft white cards, and teal highlights.
              </p>
            </div>

            {/* System Card */}
            <div
              onClick={() => setThemeMode('system')}
              className={`p-4 rounded-xl border cursor-pointer transition-all flex flex-col justify-between gap-3 ${
                themeMode === 'system'
                  ? 'border-teal-500 bg-teal-500/5 ring-2 ring-teal-500/20'
                  : 'dark:bg-navy-950/60 bg-slate-50 dark:border-navy-800 border-slate-200 hover:border-slate-300 dark:hover:border-navy-700'
              }`}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Laptop className="w-4 h-4 text-indigo-400" />
                  <span className="text-xs font-bold dark:text-coolwhite text-slate-800">System Preference</span>
                </div>
                {themeMode === 'system' && <CheckCircle2 className="w-4 h-4 text-teal-500" />}
              </div>
              <div className="h-14 rounded-lg overflow-hidden border dark:border-[#1E2E48] border-[#CBD5E1] flex">
                <div className="w-1/2 bg-[#071426] p-1.5 flex flex-col justify-between">
                  <div className="w-2.5 h-2.5 rounded-full bg-teal-500"></div>
                  <div className="w-full h-2 rounded bg-[#101F35]"></div>
                </div>
                <div className="w-1/2 bg-[#F8FAFC] p-1.5 flex flex-col justify-between">
                  <div className="w-2.5 h-2.5 rounded-full bg-teal-600"></div>
                  <div className="w-full h-2 rounded bg-white"></div>
                </div>
              </div>
              <p className="text-[11px] dark:text-muted text-slate-500">
                Automatically adapts to match your operating system light/dark preference.
              </p>
            </div>
          </div>
        </div>

        {/* Global Inference & Telemetry Parameters */}
        <div className="dark:bg-navy-900 bg-white border dark:border-navy-800 border-slate-200 rounded-2xl p-6 shadow-sm space-y-4">
          <div className="border-b dark:border-navy-800 border-slate-200 pb-3">
            <h2 className="text-sm font-bold dark:text-coolwhite text-slate-900 flex items-center gap-2">
              <Sliders className="w-4 h-4 text-teal-500" />
              Inference & Feedback Tracking Settings
            </h2>
            <p className="text-xs dark:text-muted text-slate-500 mt-0.5">
              Default configuration for scoring timeouts, candidates, and automatic implicit dwell telemetry.
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs">
            <div>
              <label className="block font-semibold dark:text-coolwhite text-slate-800 mb-1">
                Default Top-K Candidates
              </label>
              <input
                type="number"
                min="1"
                max="50"
                value={defaultTopK}
                onChange={(e) => setDefaultTopK(e.target.value)}
                className="w-full dark:bg-navy-950 bg-slate-50 border dark:border-navy-800 border-slate-200 rounded-lg px-3 py-2 dark:text-coolwhite text-slate-900 focus:outline-none focus:border-teal-500 font-mono"
              />
            </div>

            <div>
              <label className="block font-semibold dark:text-coolwhite text-slate-800 mb-1">
                Inference Timeout (ms)
              </label>
              <input
                type="number"
                min="1000"
                max="30000"
                step="500"
                value={inferenceTimeout}
                onChange={(e) => setInferenceTimeout(e.target.value)}
                className="w-full dark:bg-navy-950 bg-slate-50 border dark:border-navy-800 border-slate-200 rounded-lg px-3 py-2 dark:text-coolwhite text-slate-900 focus:outline-none focus:border-teal-500 font-mono"
              />
            </div>
          </div>

          <div className="pt-3 border-t dark:border-navy-800 border-slate-200 space-y-3">
            <label className="flex items-center gap-3 cursor-pointer">
              <input
                type="checkbox"
                checked={autoDwellTracking}
                onChange={(e) => setAutoDwellTracking(e.target.checked)}
                className="rounded dark:bg-navy-950 border-slate-300 text-teal-500 focus:ring-teal-500 w-4 h-4 cursor-pointer"
              />
              <div>
                <div className="text-xs font-semibold dark:text-coolwhite text-slate-800">
                  Automatic Dwell Time Tracking
                </div>
                <div className="text-[11px] dark:text-muted text-slate-500">
                  Logs implicit VIEW and LONG_VIEW events after 1.5s and 4.0s of candidate dwell.
                </div>
              </div>
            </label>

            <label className="flex items-center gap-3 cursor-pointer">
              <input
                type="checkbox"
                checked={enableTelemetry}
                onChange={(e) => setEnableTelemetry(e.target.checked)}
                className="rounded dark:bg-navy-950 border-slate-300 text-teal-500 focus:ring-teal-500 w-4 h-4 cursor-pointer"
              />
              <div>
                <div className="text-xs font-semibold dark:text-coolwhite text-slate-800">
                  Live Telemetry Health Polling
                </div>
                <div className="text-[11px] dark:text-muted text-slate-500">
                  Regularly checks backend health status at `/api/health`.
                </div>
              </div>
            </label>
          </div>
        </div>

        {/* Backend & Deployment Status */}
        <div className="dark:bg-navy-900 bg-white border dark:border-navy-800 border-slate-200 rounded-2xl p-6 shadow-sm space-y-4">
          <div className="border-b dark:border-navy-800 border-slate-200 pb-3 flex items-center justify-between">
            <h2 className="text-sm font-bold dark:text-coolwhite text-slate-900 flex items-center gap-2">
              <Shield className="w-4 h-4 text-teal-500" />
              Connected Backend & Environment
            </h2>
            <span className="px-2 py-0.5 rounded text-[11px] font-mono font-semibold bg-teal-500/10 text-teal-600 dark:text-teal-300 border border-teal-500/20">
              FastAPI v{health.data?.version || '1.0.0'}
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
            <div className="p-3 rounded-lg dark:bg-navy-950 bg-slate-50 border dark:border-navy-800 border-slate-200">
              <span className="dark:text-muted text-slate-500">Backend Status</span>
              <div className="font-bold dark:text-coolwhite text-slate-800 mt-1 flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-teal-500"></span>
                {health.healthy ? 'Connected & Healthy' : 'Offline'}
              </div>
            </div>

            <div className="p-3 rounded-lg dark:bg-navy-950 bg-slate-50 border dark:border-navy-800 border-slate-200">
              <span className="dark:text-muted text-slate-500">Registered Domains</span>
              <div className="font-bold dark:text-coolwhite text-slate-800 mt-1">
                {clients.length} Active Workspaces
              </div>
            </div>

            <div className="p-3 rounded-lg dark:bg-navy-950 bg-slate-50 border dark:border-navy-800 border-slate-200">
              <span className="dark:text-muted text-slate-500">Selected Scope</span>
              <div className="font-bold dark:text-coolwhite text-slate-800 mt-1 font-mono">
                {selectedClientId}
              </div>
            </div>
          </div>
        </div>

        {/* Submit */}
        <div className="flex items-center justify-end gap-3">
          <button
            type="submit"
            className="px-5 py-2.5 rounded-xl bg-teal-500 hover:bg-teal-400 text-navy-950 font-bold text-xs shadow-sm flex items-center gap-2 transition-all cursor-pointer"
          >
            <Save className="w-4 h-4" />
            Save Settings
          </button>
        </div>
      </form>
    </div>
  );
}
