Before diving into thhe aspect of multiplication through machine learning, first concept to be aware of is tensor multiplication.
A tensor multiplication tells us about the interaction between a  vector space and the other vector space, it is not a union necessarily,
but the permutation and combination of one plane to the other. By interaction we mean that the effect of one variable is depending upon another variable.
For example, we have two vector spaces, Ka and Kb, then Ka * Kb = Span{u*v  | u belongs to Ka and v belongs to Kb}, basically
forming every possible interaction that could be within the two, * here means any operation the two vector spaces have a closure on.
Example:
(a+bi)*(c+di) = (ac-bd)+(ad + bc)i
If we translate this into pairs, this becomes: (a,b)*(c,d) = (ac-bd, ad+bc)
Can be represented as:
([a b] [[1 0],[0 -1]] [c, d]), [a b][[0 1], [1 0]][c, d]
This represents the product in its entirety.