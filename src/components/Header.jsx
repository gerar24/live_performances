const Header = () => {
  return (
    <header className="py-6 border-b border-neutral-800 mb-6">
      <div className="max-w-6xl mx-auto px-4 flex justify-between items-center">
        <h1 className="text-2xl font-semibold">Monitor de Performance</h1>
        <span className="text-sm text-neutral-400">Auto-actualizado diariamente (cierre de mercado Londres 1pm ARG)</span>
      </div>
    </header>
  )
}

export default Header


