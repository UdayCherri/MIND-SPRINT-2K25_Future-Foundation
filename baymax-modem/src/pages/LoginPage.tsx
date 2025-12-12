import { motion } from 'motion/react';
import { Activity, Mail, Lock, ChevronRight, AlertCircle } from 'lucide-react';
import { useState } from 'react';
import { useUser } from '../contexts/UserContext';

interface LoginPageProps {
  onNavigateToSignUp: () => void;
}

export function LoginPage({ onNavigateToSignUp }: LoginPageProps) {
  const [email, setEmail] = useState('demo@baymax.ai');
  const [password, setPassword] = useState('demo123');
  const [error, setError] = useState('');
  const { login, isLoading } = useUser();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    
    const success = await login(email, password);
    if (!success) {
      setError('Invalid email or password');
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-indigo-950 flex items-center justify-center p-4 overflow-hidden">
      {/* Animated Background */}
      <div className="fixed inset-0 opacity-30">
        <div className="absolute top-20 left-20 w-96 h-96 bg-blue-500/20 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute bottom-20 right-20 w-96 h-96 bg-purple-500/20 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1s' }}></div>
        <div className="absolute top-1/2 left-1/2 w-96 h-96 bg-teal-500/20 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '2s' }}></div>
      </div>

      <div className="relative z-10 w-full max-w-6xl grid grid-cols-1 lg:grid-cols-2 gap-8 items-center">
        {/* Left Side - Login Form */}
        <motion.div
          initial={{ opacity: 0, x: -50 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.8 }}
          className="w-full"
        >
          <div className="rounded-3xl bg-gradient-to-br from-slate-900/40 to-slate-800/40 backdrop-blur-xl border border-white/10 p-8 lg:p-12 shadow-2xl">
            {/* Logo */}
            <div className="flex items-center space-x-3 mb-8">
              <div className="relative">
                <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-teal-400 to-blue-500 flex items-center justify-center shadow-lg shadow-teal-500/50">
                  <Activity className="w-7 h-7 text-white" />
                </div>
                <div className="absolute inset-0 rounded-xl bg-gradient-to-br from-teal-400 to-blue-500 blur-md opacity-50 animate-pulse"></div>
              </div>
              <div>
                <div className="text-2xl tracking-tight">
                  <span className="bg-gradient-to-r from-teal-400 to-blue-400 bg-clip-text text-transparent">BAYMAX</span>
                  <span className="text-white/90">: ModeM</span>
                </div>
              </div>
            </div>

            {/* Title */}
            <h1 className="text-3xl mb-2 text-white">Welcome Back</h1>
            <p className="text-white/60 mb-8">Your AI companion for cognitive and emotional wellness</p>

            {/* Demo Credentials Info */}
            <div className="mb-6 px-4 py-3 rounded-xl bg-blue-500/10 border border-blue-500/30">
              <p className="text-sm text-blue-400">
                Demo: <span className="font-mono">demo@baymax.ai</span> / <span className="font-mono">demo123</span>
              </p>
            </div>

            {/* Error Message */}
            {error && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                className="px-4 py-3 rounded-xl bg-red-500/10 border border-red-500/30 flex items-center space-x-2"
              >
                <AlertCircle className="w-5 h-5 text-red-400" />
                <p className="text-sm text-red-400">{error}</p>
              </motion.div>
            )}

            {/* Form */}
            <form onSubmit={handleSubmit} className="space-y-5">
              {/* Email Field */}
              <div>
                <label className="block text-sm text-white/70 mb-2">Email Address</label>
                <div className="relative">
                  <Mail className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-white/40" />
                  <input
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    placeholder="your.email@example.com"
                    className="w-full pl-12 pr-4 py-3 rounded-xl bg-slate-800/50 border border-white/10 text-white placeholder-white/40 focus:outline-none focus:border-teal-400/50 focus:ring-2 focus:ring-teal-400/20 transition-all"
                    required
                  />
                </div>
              </div>

              {/* Password Field */}
              <div>
                <label className="block text-sm text-white/70 mb-2">Password</label>
                <div className="relative">
                  <Lock className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-white/40" />
                  <input
                    type="password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder="Enter your password"
                    className="w-full pl-12 pr-4 py-3 rounded-xl bg-slate-800/50 border border-white/10 text-white placeholder-white/40 focus:outline-none focus:border-teal-400/50 focus:ring-2 focus:ring-teal-400/20 transition-all"
                    required
                  />
                </div>
              </div>

              {/* Login Button */}
              <motion.button
                type="submit"
                disabled={isLoading}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                className="w-full px-6 py-4 rounded-xl bg-gradient-to-r from-teal-500 to-blue-500 text-white shadow-lg shadow-teal-500/30 hover:shadow-teal-500/50 transition-all flex items-center justify-center space-x-2 relative overflow-hidden group mt-6 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <div className="absolute inset-0 bg-gradient-to-r from-teal-400 to-blue-400 opacity-0 group-hover:opacity-100 transition-opacity"></div>
                <span className="relative z-10">{isLoading ? 'Logging in...' : 'Log In'}</span>
                {!isLoading && <ChevronRight className="w-5 h-5 relative z-10" />}
              </motion.button>
            </form>

            {/* Sign Up Link */}
            <div className="mt-6 text-center">
              <span className="text-white/60">Don't have an account? </span>
              <button
                onClick={onNavigateToSignUp}
                className="text-teal-400 hover:text-teal-300 transition-colors"
              >
                Create an Account
              </button>
            </div>
          </div>
        </motion.div>

        {/* Right Side - Illustration */}
        <motion.div
          initial={{ opacity: 0, x: 50 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="hidden lg:flex items-center justify-center"
        >
          <div className="relative w-full max-w-md aspect-square">
            {/* Central Brain Illustration */}
            <div className="absolute inset-0 flex items-center justify-center">
              <motion.div
                animate={{
                  scale: [1, 1.05, 1],
                  rotate: [0, 5, -5, 0],
                }}
                transition={{
                  duration: 6,
                  repeat: Infinity,
                  ease: "easeInOut",
                }}
                className="relative w-64 h-64"
              >
                {/* Glow Effect */}
                <div className="absolute inset-0 bg-gradient-to-br from-teal-400/20 to-blue-500/20 rounded-full blur-3xl"></div>
                
                {/* Brain Network */}
                <svg className="w-full h-full" viewBox="0 0 200 200">
                  <defs>
                    <linearGradient id="brainGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                      <stop offset="0%" stopColor="#39E6C6" />
                      <stop offset="50%" stopColor="#3F77FF" />
                      <stop offset="100%" stopColor="#9B5CFF" />
                    </linearGradient>
                  </defs>
                  
                  {/* Neural Connections */}
                  <motion.circle
                    cx="100"
                    cy="100"
                    r="80"
                    fill="none"
                    stroke="url(#brainGradient)"
                    strokeWidth="2"
                    initial={{ pathLength: 0 }}
                    animate={{ pathLength: 1 }}
                    transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                    opacity="0.3"
                  />
                  <motion.circle
                    cx="100"
                    cy="100"
                    r="60"
                    fill="none"
                    stroke="url(#brainGradient)"
                    strokeWidth="2"
                    initial={{ pathLength: 0 }}
                    animate={{ pathLength: 1 }}
                    transition={{ duration: 2.5, repeat: Infinity, ease: "linear", delay: 0.3 }}
                    opacity="0.4"
                  />
                  <motion.circle
                    cx="100"
                    cy="100"
                    r="40"
                    fill="none"
                    stroke="url(#brainGradient)"
                    strokeWidth="2"
                    initial={{ pathLength: 0 }}
                    animate={{ pathLength: 1 }}
                    transition={{ duration: 3, repeat: Infinity, ease: "linear", delay: 0.6 }}
                    opacity="0.5"
                  />
                  
                  {/* Neural Nodes */}
                  {[0, 60, 120, 180, 240, 300].map((angle, i) => {
                    const x = 100 + 70 * Math.cos((angle * Math.PI) / 180);
                    const y = 100 + 70 * Math.sin((angle * Math.PI) / 180);
                    return (
                      <motion.circle
                        key={i}
                        cx={x}
                        cy={y}
                        r="4"
                        fill="#39E6C6"
                        animate={{
                          scale: [1, 1.5, 1],
                          opacity: [0.6, 1, 0.6],
                        }}
                        transition={{
                          duration: 2,
                          repeat: Infinity,
                          delay: i * 0.2,
                        }}
                        className="drop-shadow-[0_0_8px_rgba(57,230,198,0.8)]"
                      />
                    );
                  })}
                  
                  {/* Center Core */}
                  <circle cx="100" cy="100" r="15" fill="url(#brainGradient)" opacity="0.8" />
                </svg>
              </motion.div>
            </div>

            {/* Floating Particles */}
            {[...Array(8)].map((_, i) => (
              <motion.div
                key={i}
                className="absolute w-2 h-2 rounded-full bg-teal-400"
                style={{
                  top: `${Math.random() * 100}%`,
                  left: `${Math.random() * 100}%`,
                }}
                animate={{
                  y: [0, -30, 0],
                  opacity: [0, 1, 0],
                }}
                transition={{
                  duration: 3 + Math.random() * 2,
                  repeat: Infinity,
                  delay: i * 0.5,
                }}
              />
            ))}
          </div>
        </motion.div>
      </div>
    </div>
  );
}
