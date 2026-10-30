# What federated FRRI reveals

This note lists what every message of the horizontal federated FRRI protocol (`fuzzyroughrules/federated/`) reveals
about the training data of a client. Training objects are never sent as a whole, but the protocol has no formal privacy
guarantee. Differential privacy is out of scope here.

Assumptions: the server and the clients follow the protocol but are curious (honest but curious). All values are
scaled with the global range, which every client receives in step 1, so a scaled value in a message can be turned back
into the raw value: `raw = scaled * (max - min) + min`.

Notation: a rule `r` was built from a training object `x` of class `d_r`. It uses some attributes, each with a relation
type (`DOMINATED`, `DOMINANT`, `INDISCERNIBLE`) and a slope. `R_r(x, u)` is the relation of an object `u` to the rule:
the minimum over the used attributes. The confidence of a rule on a set of objects is
`min_u I(R_r(x, u), [d_r = d(u)])` with the Łukasiewicz implicator, i.e. `1 - max R_r(x, u)` over the objects `u` of
other classes (or 1 if there are none).

## Message by message

### Step 1: range and classes

The client sends the per-feature minimum and maximum of its `train_local` data and its set of labels.

- The minimum and the maximum are exact values of training objects (possibly of different objects). Outliers are
  revealed exactly.
- The set of labels tells which classes the client has. In a non-IID setting this is a strong signal about the client.

### Step 2: local rules (`RulePayload`)

For every local rule: the indices of the used attributes, the values of the training object `x` on these attributes,
their relation types and slopes, the class label, and the local confidence.

- **The values are real attribute values of a real training object**, with its class label. Only the unused attributes
  are hidden (`tests/test_federated_units.py::test_payload_contains_only_used_attributes` checks that). A rule that
  uses all attributes reveals the whole object.
- **The local confidence reveals a distance to another object.** FRRI stores the lower approximation membership of `x`,
  computed with all attributes, `INDISCERNIBLE` and slope 1. That equals `min(1, D)`, where `D` is the L-infinity
  distance (in scaled units) from `x` to the nearest object of another class on the same client. This was checked on
  wine, breast cancer and synthetic data with 2 and 5 clients: the largest difference between the stored confidence and
  `min(1, D)` was 5.6e-17. So the client also reveals that one of its objects of another class lies on the boundary of
  the L-infinity box with radius `D` around `x`, and none lies inside it.
- After step 3, the rules are broadcast to **all clients**, not only to the server. Every client sees the rule payloads
  of every other client.
- The local confidence is only needed for variant (a) and for the stopping rule of rule repair. The other variants use
  the global confidence and could leave it out.

### Step 3: global confidence

For every candidate rule, the client sends its local minimum `1 - max R_r(x, u)` over its objects `u` of other classes.

- Every rule of another client acts as a probe. The answer tells how close the nearest object of another class on this
  client comes to the rule. For a rule with a single attribute and relation `INDISCERNIBLE` with slope `s`, it gives
  `|x_a - u_a| = s * (1 - R)`, i.e. the value of that attribute of the nearest object of another class up to its sign.
  With `DOMINATED` or `DOMINANT` the sign is known too. More attributes give a level set instead of a point.
- A minimum of 1 reveals that no object of another class has a positive relation to the rule.
- The server sees one number per (rule, client), so it also learns which client limits each rule. A secure minimum
  (secure aggregation) would only reveal the global minimum.

### Step 4: coverage for the set cover

The client sends a binary matrix (rules x local objects), with the columns in random order.

- The number of columns is the number of `train_local` objects of the client.
- **Every covered column reveals the class of its object.** With the global confidence, a rule can not cover an object
  of another class: such an object has `R_r(x, u) <= 1 - confidence`, so its membership `max(R + confidence - 1, 0)`
  is 0. This was checked on the same datasets: no covered object was covered by a rule of another class. So the
  matrix reveals the class distribution of the covered objects of each client.
- The pattern of a column tells in which rule regions the object lies. With many rules this places the object in a
  small cell of the feature space. Shuffling the columns hides their order, not these patterns.
- In coverage rounds the columns are shuffled again in every round (another seed), but the same object gives the same
  pattern for the same rules, so the columns of different rounds can be matched.

### Step 5, only for variant (c) ranked by coverage: coverage counts

One integer per own rule: how many local objects the rule covers with its local confidence. This reveals the density of
the client's objects of that class around the rule.

### Rule repair (new round)

While the global confidence of a rule is below its local confidence, its owner sends the rule with one more attribute,
and every client returns a minimum for the longer rule.

- Every round reveals one more attribute value of the training object of every rule that is still being repaired.
- The diagnosis in `docs/federated-improvements-report.md` (item B) shows that even the unshortened rule usually loses
  confidence on the data of the other clients, so most rules never recover and end up with **all attributes, i.e. the
  complete training object**. In the run `python -m experiments.horizontal_fl_frri --loader builtin --data-dir wine
  --clients 3 --seed 0`, 12 of the 13 rules needed repair, 3 recovered, and 9 ended with all 13 attributes. The
  number of revealed attribute values went from 48 to 138.
- Every round also gives new probes for step 3, and it adds rounds: `repair_n_rounds` in the output of the experiment.
- `--repair-max-rounds` limits the number of extra attribute values per rule.

### Coverage rounds (new round)

After a set cover, every client sends the FRRI rules of its objects that the selected rules do not cover.

- Every new candidate is another training object (its used attributes and its label) that is revealed, and it tells
  that this object was not covered. The experiment logs this as `n_extra_rules_exposed`.
- Every round repeats steps 3 and 4 for the new rules.

### Not part of the protocol

The default-rule baseline (`FallbackRuleGenerator` with `fallback="default"`) needs the class counts of the clients,
which the protocol does not send. The fallback `"relation"` needs nothing beyond the rules.

## Options to reveal less

None of these are implemented. A useful property of the protocol is that the global confidence is computed by the
clients for exactly the rule that is sent. A client can therefore change a rule before sending it, and the global
confidence stays correct for the changed rule. Only the claim that a rule has the local confidence of its object is
lost.

- **Rounding.** Send the values on a coarse grid, e.g. 0.05 in scaled units. The rule no longer contains the exact
  values of the object; the confidence is computed for the rounded rule. The cost is a lower confidence, because the
  rounded rule can come closer to objects of other classes.
- **Intervals.** Send an interval of values that contains the value of the object and enough other local objects
  (similar to k-anonymity), with a relation that is 1 inside the interval and decreases outside it. This needs a new
  relation type in `fuzzyroughrules/operators.py`.
- **Public range.** Use a range that is agreed beforehand from domain knowledge instead of the minimum and maximum of
  the data, or round the empirical range outwards.
- **No local confidence.** Leave it out of the payload when the variant does not need it (all variants except (a) and
  rule repair).
- **Secure aggregation of the minima.** The server only learns the global minimum of each rule, not which client gives
  it.
- **Counts instead of coverage patterns.** A greedy set cover only needs, in every round, the number of still uncovered
  objects that each candidate covers, summed over the clients (possibly with secure aggregation). This reveals counts
  instead of the patterns of single objects, and the labels of covered objects only in aggregate. It needs one round
  per selected rule, and greedy set cover is not optimal.
- **No coverage counts.** Rank by global confidence in variant (c), which needs no extra message.
- **Limits on the new rounds.** Limit the number of repair rounds and of coverage rounds; both are options of the
  experiment.
