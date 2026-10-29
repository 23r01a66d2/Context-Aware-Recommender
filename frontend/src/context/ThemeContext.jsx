import React, { createContext, useContext, useState, useEffect } from 'react';

const ThemeContext = createContext();

const defaultThemeContext = {
  themeMode: 'dark',
  resolvedTheme: 'dark',
  setThemeMode: () => {},
  toggleTheme: () => {},
};

export function ThemeProvider({ children }) {
  const [themeMode, setThemeModeState] = useState(() => {
    try {
      return localStorage.getItem('app_theme_mode') || 'dark';
    } catch {
      return 'dark';
    }
  });

  const [systemIsDark, setSystemIsDark] = useState(() => {
    if (typeof window !== 'undefined' && window.matchMedia) {
      return window.matchMedia('(prefers-color-scheme: dark)').matches;
    }
    return true;
  });

  useEffect(() => {
    if (typeof window === 'undefined' || !window.matchMedia) return;
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    const handler = (e) => setSystemIsDark(e.matches);
    mediaQuery.addEventListener('change', handler);
    return () => mediaQuery.removeEventListener('change', handler);
  }, []);

  const resolvedTheme = themeMode === 'system' ? (systemIsDark ? 'dark' : 'light') : themeMode;

  useEffect(() => {
    const root = document.documentElement;
    root.classList.remove('dark', 'light', 'minimal');
    if (resolvedTheme === 'dark') {
      root.classList.add('dark');
      root.setAttribute('data-theme', 'dark');
      root.style.colorScheme = 'dark';
    } else if (resolvedTheme === 'minimal') {
      root.classList.add('minimal');
      root.setAttribute('data-theme', 'minimal');
      root.style.colorScheme = 'light';
    } else {
      root.classList.add('light');
      root.setAttribute('data-theme', 'light');
      root.style.colorScheme = 'light';
    }
  }, [resolvedTheme]);

  const setThemeMode = (mode) => {
    setThemeModeState(mode);
    try {
      localStorage.setItem('app_theme_mode', mode);
    } catch (e) {
      console.warn('Failed to persist theme to localStorage:', e);
    }
  };

  const toggleTheme = () => {
    const nextMode = resolvedTheme === 'dark' ? 'light' : resolvedTheme === 'light' ? 'minimal' : 'dark';
    setThemeMode(nextMode);
  };

  return (
    <ThemeContext.Provider value={{ themeMode, resolvedTheme, setThemeMode, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  const context = useContext(ThemeContext);
  return context || defaultThemeContext;
}
