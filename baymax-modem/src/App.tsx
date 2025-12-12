import { useState } from 'react';
import { LoginPage } from './pages/LoginPage';
import { SignUpPage } from './pages/SignUpPage';
import { DashboardPage } from './pages/DashboardPage';
import { AnalyticsPage } from './pages/AnalyticsPage';
import { InsightsPage } from './pages/InsightsPage';
import { SettingsPage } from './pages/SettingsPage';
import { ErrorBoundary } from './components/ErrorBoundary';
import { UserProvider, useUser } from './contexts/UserContext';

export type Page = 'login' | 'signup' | 'dashboard' | 'analytics' | 'insights' | 'settings';

function AppContent() {
  const [currentPage, setCurrentPage] = useState<Page>('login');
  const { user, logout } = useUser();

  const handleLogout = () => {
    logout();
    setCurrentPage('login');
  };

  const renderPage = () => {
    if (!user) {
      if (currentPage === 'signup') {
        return <SignUpPage onNavigateToLogin={() => setCurrentPage('login')} />;
      }
      return <LoginPage onNavigateToSignUp={() => setCurrentPage('signup')} />;
    }

    switch (currentPage) {
      case 'dashboard':
        return <DashboardPage onNavigate={setCurrentPage} onLogout={handleLogout} />;
      case 'analytics':
        return <AnalyticsPage onNavigate={setCurrentPage} onLogout={handleLogout} />;
      case 'insights':
        return <InsightsPage onNavigate={setCurrentPage} onLogout={handleLogout} />;
      case 'settings':
        return <SettingsPage onNavigate={setCurrentPage} onLogout={handleLogout} />;
      default:
        return <DashboardPage onNavigate={setCurrentPage} onLogout={handleLogout} />;
    }
  };

  return <ErrorBoundary>{renderPage()}</ErrorBoundary>;
}

export default function App() {
  return (
    <UserProvider>
      <AppContent />
    </UserProvider>
  );
}
