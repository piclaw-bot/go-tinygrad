package tensor

// UPat is a pattern for matching UOp nodes in graph rewriting.
type UPat struct {
	Ops      []Ops  // match any of these ops (nil = any)
	Name     string // capture name (empty = don't capture)
	Src      []*UPat // child patterns (nil = don't check children)
	ArgMatch func(any) bool // optional arg predicate
}

// Pat creates a pattern matching specific ops.
func Pat(ops ...Ops) *UPat { return &UPat{Ops: ops} }

// Named returns a copy with a capture name.
func (p *UPat) Named(name string) *UPat {
	c := *p
	c.Name = name
	return &c
}

// WithSrc sets child patterns.
func (p *UPat) WithSrc(src ...*UPat) *UPat {
	c := *p
	c.Src = src
	return &c
}

// AnyOp matches any operation.
func AnyOp() *UPat { return &UPat{} }

// Var creates a wildcard pattern with a capture name.
func Var(name string) *UPat { return &UPat{Name: name} }

// ConstPat matches a constant op.
func ConstPat(name string) *UPat { return &UPat{Ops: []Ops{OpConst}, Name: name} }

// Match attempts to match this pattern against a UOp, returning captured bindings.
func (p *UPat) Match(u *UOp) (map[string]*UOp, bool) {
	bindings := map[string]*UOp{}
	if p.match(u, bindings) {
		return bindings, true
	}
	return nil, false
}

func (p *UPat) match(u *UOp, bindings map[string]*UOp) bool {
	// Check op
	if p.Ops != nil {
		found := false
		for _, op := range p.Ops {
			if u.Op == op { found = true; break }
		}
		if !found { return false }
	}

	// Check arg
	if p.ArgMatch != nil && !p.ArgMatch(u.Arg) {
		return false
	}

	// Check children
	if p.Src != nil {
		if len(p.Src) != len(u.Src) {
			return false
		}
		for i, sp := range p.Src {
			if !sp.match(u.Src[i], bindings) {
				return false
			}
		}
	}

	// Capture
	if p.Name != "" {
		if existing, ok := bindings[p.Name]; ok {
			if existing != u { return false } // same name must bind same node
		}
		bindings[p.Name] = u
	}
	return true
}
