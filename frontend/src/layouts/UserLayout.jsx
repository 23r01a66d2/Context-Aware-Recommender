import React from 'react';
import { Outlet } from 'react-router-dom';
import Navbar from '../components/Navbar';

export default function UserLayout() {
  return (
    <div className="min-h-screen dark:bg-navy-950 bg-slate-50 dark:text-coolwhite text-slate-800 flex flex-col font-sans transition-colors duration-200">
      <Navbar />
      <main className="flex-1 max-w-7xl mx-auto w-full p-4 sm:p-6 lg:p-8">
        <Outlet />
      </main>
    </div>
  );
}
