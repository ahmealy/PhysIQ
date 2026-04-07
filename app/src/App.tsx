import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Sidebar } from './components/Sidebar';
import { Dashboard } from './pages/Dashboard';
import { Train } from './pages/Train';
import { Predict } from './pages/Predict';
import { Visualize } from './pages/Visualize';
import { DatasetStudio } from './pages/DatasetStudio';
import { PipelineView } from './pages/PipelineView';
import { ExperimentTracking } from './pages/ExperimentTracking';

export default function App() {
  return (
    <Router>
      <div className="flex min-h-screen bg-slate-950 text-slate-200 font-sans selection:bg-blue-500/30 selection:text-blue-200">
        <Sidebar />
        <main className="flex-1 overflow-y-auto bg-[radial-gradient(ellipse_at_top_right,_var(--tw-gradient-stops))] from-slate-900 via-slate-950 to-slate-950">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/train" element={<Train />} />
            <Route path="/predict" element={<Predict />} />
            <Route path="/visualize" element={<Visualize />} />
            <Route path="/dataset" element={<DatasetStudio />} />
            <Route path="/pipeline" element={<PipelineView />} />
            <Route path="/experiments" element={<ExperimentTracking />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}
