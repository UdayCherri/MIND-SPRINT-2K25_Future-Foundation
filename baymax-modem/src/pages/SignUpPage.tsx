import { motion } from 'motion/react';
import { Activity, Mail, Lock, User, ChevronRight, Heart, AlertCircle } from 'lucide-react';
import { useState } from 'react';
import { useUser } from '../contexts/UserContext';

interface SignUpPageProps {
  onNavigateToLogin: () => void;
}

export function SignUpPage({ onNavigateToLogin }: SignUpPageProps) {
  const [fullName, setFullName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState('');
  const { register, isLoading } = useUser();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    
    if (password !== confirmPassword) {
      setError('Passwords do not match');
      return;
    }
    
    if (password.length < 6) {
      setError('Password must be at least 6 characters');
      return;
    }
    
    const success = await register(email, password, fullName);
    if (!success) {
      setError('Email already exists or registration failed');
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
        {/* Left Side - Sign Up Form */}
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
            <h1 className="text-3xl mb-2 text-white">Create Your Account</h1>
            <p className="text-white/60 mb-8">Begin your journey to better mental wellness</p>

            {/* Form */}
            <form onSubmit={handleSubmit} className="space-y-4">
              {/* Full Name Field */}
              <div>
                <label className="block text-sm text-white/70 mb-2">Full Name</label>
                <div className="relative">
                  <User className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-white/40" />
                  <input
                    type="text"
                    value={fullName}
                    onChange={(e) => setFullName(e.target.value)}
                    placeholder="John Doe"
                    className="w-full pl-12 pr-4 py-3 rounded-xl bg-slate-800/50 border border-white/10 text-white placeholder-white/40 focus:outline-none focus:border-teal-400/50 focus:ring-2 focus:ring-teal-400/20 transition-all"
                    required
                  />
                </div>
              </div>

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
                    placeholder="Create a password"
                    className="w-full pl-12 pr-4 py-3 rounded-xl bg-slate-800/50 border border-white/10 text-white placeholder-white/40 focus:outline-none focus:border-teal-400/50 focus:ring-2 focus:ring-teal-400/20 transition-all"
                    required
                  />
                </div>
              </div>

              {/* Confirm Password Field */}
              <div>
                <label className="block text-sm text-white/70 mb-2">Confirm Password</label>
                <div className="relative">
                  <Lock className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-white/40" />
                  <input
                    type="password"
                    value={confirmPassword}
                    onChange={(e) => setConfirmPassword(e.target.value)}
                    placeholder="Confirm your password"
                    className="w-full pl-12 pr-4 py-3 rounded-xl bg-slate-800/50 border border-white/10 text-white placeholder-white/40 focus:outline-none focus:border-teal-400/50 focus:ring-2 focus:ring-teal-400/20 transition-all"
                    required
                  />
                </div>
              </div>

              {/* MBTI Field (Optional) */}
              <div>
                <label className="block text-sm text-white/70 mb-2">MBTI Type (Optional)</label>
                <select
                  value={mbti}
                  onChange={(e) => setMbti(e.target.value)}
                  className="w-full px-4 py-3 rounded-xl bg-slate-800/50 border border-white/10 text-white focus:outline-none focus:border-teal-400/50 focus:ring-2 focus:ring-teal-400/20 transition-all cursor-pointer"
                >
                  <option value="">Select your MBTI type</option>
                  {mbtiTypes.map((type) => (
                    <option key={type} value={type}>{type}</option>
                  ))}
                </select>
              </div>

              {/* Sign Up Button */}
              <motion.button
                type="submit"
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                className="w-full px-6 py-4 rounded-xl bg-gradient-to-r from-teal-500 to-blue-500 text-white shadow-lg shadow-teal-500/30 hover:shadow-teal-500/50 transition-all flex items-center justify-center space-x-2 relative overflow-hidden group mt-6"
              >
                <div className="absolute inset-0 bg-gradient-to-r from-teal-400 to-blue-400 opacity-0 group-hover:opacity-100 transition-opacity"></div>
                <span className="relative z-10">Create Account</span>
                <ChevronRight className="w-5 h-5 relative z-10" />
              </motion.button>
            </form>

            {/* Login Link */}
            <div className="mt-6 text-center">
              <span className="text-white/60">Already have an account? </span>
              <button
                onClick={onNavigateToLogin}
                className="text-teal-400 hover:text-teal-300 transition-colors"
              >
                Log In
              </button>
            </div>
          </div>
        </motion.div>

        {/* Right Side - Wellness Illustration */}
        <motion.div
          initial={{ opacity: 0, x: 50 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="hidden lg:flex items-center justify-center"
        >
          <div className="relative w-full max-w-md aspect-square">
            {/* Central Wellness Icon */}
            <div className="absolute inset-0 flex items-center justify-center">
              <motion.div
                animate={{
                  scale: [1, 1.05, 1],
                }}
                transition={{
                  duration: 4,
                  repeat: Infinity,
                  ease: "easeInOut",
                }}
                className="relative w-64 h-64"
              >
                {/* Glow Effect */}
                <div className="absolute inset-0 bg-gradient-to-br from-purple-400/20 to-pink-500/20 rounded-full blur-3xl"></div>
                
                {/* BPM Wave Pattern */}
                <svg className="w-full h-full" viewBox="0 0 200 200">
                  <defs>
                    <linearGradient id="waveGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                      <stop offset="0%" stopColor="#39E6C6" />
                      <stop offset="50%" stopColor="#9B5CFF" />
                      <stop offset="100%" stopColor="#3F77FF" />
                    </linearGradient>
                  </defs>
                  
                  {/* Heart Rate Wave */}
                  <motion.path
                    d="M 20,100 L 60,100 L 70,60 L 80,140 L 90,100 L 180,100"
                    fill="none"
                    stroke="url(#waveGradient)"
                    strokeWidth="3"
                    strokeLinecap="round"
                    initial={{ pathLength: 0, opacity: 0 }}
                    animate={{ pathLength: 1, opacity: 1 }}
                    transition={{
                      pathLength: { duration: 2, repeat: Infinity, ease: "linear" },
                      opacity: { duration: 0.5 }
                    }}
                    className="drop-shadow-[0_0_8px_rgba(57,230,198,0.6)]"
                  />
                  
                  {/* Center Heart Icon */}
                  <motion.g
                    animate={{
                      scale: [1, 1.1, 1],
                    }}
                    transition={{
                      duration: 1.5,
                      repeat: Infinity,
                      ease: "easeInOut",
                    }}
                  >
                    <circle cx="100" cy="100" r="30" fill="url(#waveGradient)" opacity="0.2" />
                    <circle cx="100" cy="100" r="20" fill="url(#waveGradient)" opacity="0.4" />
                  </motion.g>
                  
                  {/* Orbiting Particles */}
                  {[0, 90, 180, 270].map((angle, i) => {
                    const radius = 60;
                    return (
                      <motion.circle
                        key={i}
                        cx="100"
                        cy="100"
                        r="3"
                        fill="#39E6C6"
                        animate={{
                          cx: 100 + radius * Math.cos((angle * Math.PI) / 180),
                          cy: 100 + radius * Math.sin((angle * Math.PI) / 180),
                          opacity: [0.4, 1, 0.4],
                        }}
                        transition={{
                          duration: 4,
                          repeat: Infinity,
                          ease: "linear",
                          delay: i * 0.25,
                        }}
                        className="drop-shadow-[0_0_4px_rgba(57,230,198,0.8)]"
                      />
                    );
                  })}
                </svg>

                {/* Heart Icon Overlay */}
                <div className="absolute inset-0 flex items-center justify-center">
                  <Heart className="w-12 h-12 text-teal-400 fill-teal-400/20" />
                </div>
              </motion.div>
            </div>

            {/* Floating Elements */}
            {[...Array(6)].map((_, i) => (
              <motion.div
                key={i}
                className="absolute w-2 h-2 rounded-full bg-purple-400"
                style={{
                  top: `${20 + Math.random() * 60}%`,
                  left: `${20 + Math.random() * 60}%`,
                }}
                animate={{
                  y: [0, -20, 0],
                  opacity: [0, 1, 0],
                  scale: [0, 1, 0],
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
