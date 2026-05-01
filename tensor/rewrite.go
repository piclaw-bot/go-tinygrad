package tensor

// RewriteRule pairs a pattern with a rewrite function.
// The function receives the matched bindings and returns a replacement UOp, or nil to skip.
type RewriteRule struct {
	Pat     *UPat
	Rewrite func(bindings map[string]*UOp, matched *UOp) *UOp
}

// PatternMatcher holds rewrite rules indexed by op for fast dispatch.
type PatternMatcher struct {
	rules   []RewriteRule
	byOp    map[Ops][]int // op → indices into rules
	anyRules []int         // rules with nil Ops (match any)
}

// NewPatternMatcher creates a pattern matcher from a set of rules.
func NewPatternMatcher(rules ...RewriteRule) *PatternMatcher {
	pm := &PatternMatcher{
		rules: rules,
		byOp:  map[Ops][]int{},
	}
	for i, r := range rules {
		if r.Pat.Ops == nil {
			pm.anyRules = append(pm.anyRules, i)
		} else {
			for _, op := range r.Pat.Ops {
				pm.byOp[op] = append(pm.byOp[op], i)
			}
		}
	}
	return pm
}

// Rewrite tries all matching rules on a UOp and returns the first successful rewrite.
func (pm *PatternMatcher) Rewrite(u *UOp) *UOp {
	// Try op-specific rules first
	if indices, ok := pm.byOp[u.Op]; ok {
		for _, i := range indices {
			if result := pm.tryRule(i, u); result != nil {
				return result
			}
		}
	}
	// Try any-op rules
	for _, i := range pm.anyRules {
		if result := pm.tryRule(i, u); result != nil {
			return result
		}
	}
	return nil
}

func (pm *PatternMatcher) tryRule(i int, u *UOp) *UOp {
	r := &pm.rules[i]
	bindings, ok := r.Pat.Match(u)
	if !ok {
		return nil
	}
	return r.Rewrite(bindings, u)
}

// GraphRewrite applies a PatternMatcher bottom-up to the entire UOp DAG reachable from root.
// Returns the rewritten root (may be the same pointer if nothing changed).
func GraphRewrite(root *UOp, pm *PatternMatcher) *UOp {
	replaced := map[*UOp]*UOp{}
	return graphRewriteNode(root, pm, replaced)
}

func graphRewriteNode(u *UOp, pm *PatternMatcher, replaced map[*UOp]*UOp) *UOp {
	if r, ok := replaced[u]; ok {
		return r
	}

	// Bottom-up: rewrite children first
	newSrc := make([]*UOp, len(u.Src))
	changed := false
	for i, s := range u.Src {
		ns := graphRewriteNode(s, pm, replaced)
		newSrc[i] = ns
		if ns != s {
			changed = true
		}
	}

	// Rebuild with new children if any changed
	node := u
	if changed {
		node = &UOp{Op: u.Op, DType: u.DType, Src: newSrc, Arg: u.Arg, buf: u.buf}
	}

	// Try rewrite rules on this node (iterate to fixed point)
	for {
		rewritten := pm.Rewrite(node)
		if rewritten == nil {
			break
		}
		// Recursively rewrite the new node (it may have new children)
		node = graphRewriteNode(rewritten, pm, replaced)
	}

	replaced[u] = node
	return node
}
