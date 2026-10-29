import React from 'react';
import { useTheme } from '../context/ThemeContext';
import { Sun, Moon, Laptop, Palette } from 'lucide-react';

export default function ThemeSwitcher({ className = '' }) {
  const { themeMode, resolvedTheme, setThemeMode } = useTheme();

  return (
    <div
      className={`inline-flex items-center p-0.5 rounded-full border transition-colors shrink-0 ${
        resolvedTheme === 'dark'
          ? 'bg-navy-900 border-navy-800 text-muted'
          : resolvedTheme === 'minimal'
          ? 'bg-white border-purple-200 text-slate-600 shadow-sm'
          : 'bg-slate-100 border-slate-200 text-slate-600'
      } ${className}`}
      role="group"
      aria-label="Theme mode switcher"
    >
      {/* Dark Button */}
      <button
        type="button"
        onClick={() => setThemeMode('dark')}
        className={`p-1.5 rounded-full transition-all cursor-pointer flex items-center justify-center ${
          themeMode === 'dark'
            ? 'bg-teal-500 text-navy-950 shadow-sm'
            : resolvedTheme === 'dark'
            ? 'hover:text-coolwhite hover:bg-navy-800'
            : 'hover:text-slate-900 hover:bg-slate-200'
        }`}
        title="Dark theme"
        aria-label="Dark theme"
      >
        <Moon className="w-3.5 h-3.5" />
      </button>

      {/* Light Button */}
      <button
        type="button"
        onClick={() => setThemeMode('light')}
        className={`p-1.5 rounded-full transition-all cursor-pointer flex items-center justify-center ${
          themeMode === 'light'
            ? 'bg-teal-500 text-white shadow-sm'
            : resolvedTheme === 'dark'
            ? 'hover:text-coolwhite hover:bg-navy-800'
            : 'hover:text-slate-900 hover:bg-slate-200'
        }`}
        title="Light theme"
        aria-label="Light theme"
      >
        <Sun className="w-3.5 h-3.5" />
      </button>

      {/* Minimal (White & Purple) Button */}
      <button
        type="button"
        onClick={() => setThemeMode('minimal')}
        className={`p-1.5 rounded-full transition-all cursor-pointer flex items-center justify-center ${
          themeMode === 'minimal'
            ? 'bg-purple-600 text-white shadow-sm'
            : resolvedTheme === 'dark'
            ? 'hover:text-coolwhite hover:bg-navy-800'
            : 'hover:text-purple-700 hover:bg-purple-50'
        }`}
        title="Minimal theme"
        aria-label="Minimal theme"
      >
        <Palette className="w-3.5 h-3.5" />
      </button>

      {/* System Button */}
      <button
        type="button"
        onClick={() => setThemeMode('system')}
        className={`p-1.5 rounded-full transition-all cursor-pointer flex items-center justify-center ${
          themeMode === 'system'
            ? resolvedTheme === 'dark'
              ? 'bg-teal-500 text-navy-950 shadow-sm'
              : 'bg-teal-500 text-white shadow-sm'
            : resolvedTheme === 'dark'
            ? 'hover:text-coolwhite hover:bg-navy-800'
            : 'hover:text-slate-900 hover:bg-slate-200'
        }`}
        title="System theme"
        aria-label="System theme"
      >
        <Laptop className="w-3.5 h-3.5" />
      </button>
    </div>
  );
}
