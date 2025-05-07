"use client"

import Link from "next/link"
import { useState } from "react"
import { usePathname } from "next/navigation"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { BookOpen, Menu, X, ChevronDown } from "lucide-react"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"
import { SearchButton } from "@/components/search"

export default function Navbar() {
  const [isMenuOpen, setIsMenuOpen] = useState(false)
  const pathname = usePathname()

  const routes = [
    { name: "Home", path: "/" },
    {
      name: "Models",
      path: "/models",
      dropdown: true,
      items: [
        { name: "All Models", path: "/models" },
        { name: "Regression", path: "/models/regression" },
        { name: "Classification", path: "/models/classification" },
        { name: "Clustering", path: "/models/clustering" },
        { name: "Dimensionality Reduction", path: "/models/dimensionality-reduction" },
        { name: "Neural Networks", path: "/models/neural-networks" },
        { name: "Comparison", path: "/models/comparison" },
      ],
    },
    { name: "About", path: "/about" },
    { name: "Resources", path: "/resources" },
  ]

  return (
    <header className="sticky top-0 z-50 w-full border-b border-neutral-300 bg-white">
      <div className="container flex h-16 items-center justify-between">
        <Link href="/" className="flex items-center space-x-2">
          <BookOpen className="h-6 w-6 text-neutral-900" />
          <span className="font-bold text-xl text-neutral-900">ML Notebook</span>
        </Link>

        {/* Desktop navigation */}
        <nav className="hidden md:flex items-center gap-6">
          {routes.map((route) =>
            route.dropdown ? (
              <DropdownMenu key={route.path}>
                <DropdownMenuTrigger className="flex items-center gap-1 text-sm font-medium transition-colors hover:text-neutral-900">
                  {route.name}
                  <ChevronDown className="h-4 w-4" />
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end">
                  {route.items?.map((item) => (
                    <DropdownMenuItem key={item.path} asChild>
                      <Link
                        href={item.path}
                        className={cn(
                          "w-full",
                          pathname === item.path ? "font-medium text-neutral-900" : "text-neutral-600",
                        )}
                      >
                        {item.name}
                      </Link>
                    </DropdownMenuItem>
                  ))}
                </DropdownMenuContent>
              </DropdownMenu>
            ) : (
              <Link
                key={route.path}
                href={route.path}
                className={cn(
                  "text-sm font-medium transition-colors hover:text-neutral-900",
                  pathname === route.path
                    ? "text-neutral-900 underline decoration-2 underline-offset-4"
                    : "text-neutral-600",
                )}
              >
                {route.name}
              </Link>
            ),
          )}
          <SearchButton />
          <Button asChild variant="notebook">
            <Link href="/models/linear-regression">Start Learning</Link>
          </Button>
        </nav>

        {/* Mobile menu button and search */}
        <div className="flex items-center gap-2 md:hidden">
          <SearchButton />
          <Button variant="ghost" size="icon" onClick={() => setIsMenuOpen(!isMenuOpen)}>
            {isMenuOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
          </Button>
        </div>
      </div>

      {/* Mobile navigation */}
      {isMenuOpen && (
        <div className="md:hidden border-b border-neutral-300">
          <div className="container py-4 space-y-4">
            {routes.map((route) =>
              route.dropdown ? (
                <div key={route.path} className="space-y-2">
                  <div className="font-medium text-neutral-900">{route.name}</div>
                  <div className="pl-4 space-y-2 border-l-2 border-neutral-200">
                    {route.items?.map((item) => (
                      <Link
                        key={item.path}
                        href={item.path}
                        className={cn(
                          "block py-1 text-sm transition-colors hover:text-neutral-900",
                          pathname === item.path ? "text-neutral-900 font-medium" : "text-neutral-600",
                        )}
                        onClick={() => setIsMenuOpen(false)}
                      >
                        {item.name}
                      </Link>
                    ))}
                  </div>
                </div>
              ) : (
                <Link
                  key={route.path}
                  href={route.path}
                  className={cn(
                    "block py-2 text-sm font-medium transition-colors hover:text-neutral-900",
                    pathname === route.path
                      ? "text-neutral-900 underline decoration-2 underline-offset-4"
                      : "text-neutral-600",
                  )}
                  onClick={() => setIsMenuOpen(false)}
                >
                  {route.name}
                </Link>
              ),
            )}
            <Button asChild className="w-full" variant="notebook">
              <Link href="/models/linear-regression" onClick={() => setIsMenuOpen(false)}>
                Start Learning
              </Link>
            </Button>
          </div>
        </div>
      )}
    </header>
  )
}
