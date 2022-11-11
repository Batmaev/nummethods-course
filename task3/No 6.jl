### A Pluto.jl notebook ###
# v0.19.12

using Markdown
using InteractiveUtils

# ╔═╡ 0572e29a-5ae4-11ed-2a0d-ebfa7b174b56
begin
	using NPZ
	using LinearAlgebra
end

# ╔═╡ 652d1c8d-d6a8-4424-a136-da3ba9d6a638
data = npzread("./data_2.npz")

# ╔═╡ 9d1b650a-01b3-48e9-9328-c78e4170741b
md"""
Нужно решить систему уравнений вида

```math
\begin{align*}
(\tilde r_i - \tilde r_j) \cdot (\tilde r_i - \tilde r_k) &= |\tilde r_i - \tilde r_j| \cdot |\tilde r_i - \tilde r_k| \cos(\theta_{ijk}) \\
\Bigl[\vec r_{ij} \coloneqq r_i - r_j, \quad \vec n_{ij} &= r_{ij} / |r_{ij}|, \quad \theta = \theta_{ijk} \Bigr] \\
(r_{ij} + dr_i - dr_j) \cdot (r_{ik} + dr_i - dr_k) &= |r_{ij} + dr_i - dr_j| \cdot |r_{ij} + dr_k - dr_k| \cos\theta \\
r_{ij} \cdot r_{ik} + dr_i \cdot (r_{ik} + r_{ij}) - dr_j \cdot r_{ik} - dr_k \cdot r_{ij} 
&\approx 
\sqrt{r^2_{ij} + 2 r_{ij} \cdot (dr_i - dr_j)} \cdot \sqrt{r^2_{ik} + 2 r_{ik} \cdot (dr_i - dr_k)} \: \cos\theta \\
&\approx \bigl(|r_{ij}| + n_{ij} \cdot (dr_i - dr_j)\bigr) \cdot \bigl(|r_{ik}| + n_{ik} \cdot (dr_i - dr_k)\bigr) \cos\theta \\
&\approx \bigl(|r_{ij}||r_{ik}| + dr_i \cdot (|r_{ik}|n_{ij} + |r_{ij}|n_{ik}) - |r_{ik}|n_{ij} \cdot dr_j - |r_{ij}|n_{ik} \cdot dr_k\bigr) \cos\theta \\
\bigl[\vec \alpha = |r_{ik}|(\vec n_{ik} - \vec n_{ij}\cos\theta) &\qquad \vec \beta = |r_{ij}|(\vec n_{ij} - \vec n_{ik}\cos\theta) \bigr] \\
dr_i \cdot (\vec\alpha + \vec\beta) - dr_j \cdot \vec\alpha - dr_k \cdot \vec\beta &= |r_{ij}||r_{ik}| \cos\theta - r_{ij} \cdot r_{ik}
\end{align*}
```
"""

# ╔═╡ 6297c7a2-5964-4b8d-a532-19c6c528e65f
function make_system(r, p, theta)
	A   = zeros(length(theta), length(r))
	rhs = zeros(length(theta))
	for (θ, (i, j, k), n) ∈ zip(theta, eachrow(p), 1 : length(theta))

		r_ij = r[i, :] - r[j, :]
		ρ_ij = norm(r_ij)
		n_ij = r_ij / ρ_ij
		
		r_ik = r[i, :] - r[k, :]
		ρ_ik = norm(r_ik)
		n_ik = r_ik / ρ_ik

		α = ρ_ik * (n_ik - n_ij * cos(θ))
		β = ρ_ij * (n_ij - n_ik * cos(θ))

		A[n, 2i-1:2i] = α + β
		A[n, 2j-1:2j] = -α
		A[n, 2k-1:2k] = -β

		rhs[n] = ρ_ij * ρ_ik * cos(θ) - r_ij ⋅ r_ik
	end
	return A, rhs
end	

# ╔═╡ 6ace1ea9-2654-404e-a09b-20e72c287db2
md"""
Если система ``Ax = f`` недоопределённая, то мы вместо этой системы будем решать задачу оптимизации
```math
\min \Vert x \Vert^2 \;\text{ при }\; Ax=f
```
Функция Лагранжа:
```math
L = x^T x - \lambda^T \bigl(Ax - f\bigr)
```
Ищем стационарную точку:
```math
\begin{align}
\frac{\partial L}{\partial x} &= 2x^T - \lambda^T A = 0 \tag{1} \\
\frac{\partial L}{\partial \lambda} &= f^T - x^T A^T = 0 \tag{2}
\end{align}
```
Решаем:
```math
\lambda^T A A^T \;\overset{(1)}{=}\; 2x^T A^T \;\overset{(2)}{=}\; 2f^T
```
```math
\lambda^T = 2f^T \bigl(AA^T\bigr)^{-1}
```
```math
x^T \;\overset{(1)}{=}\; \frac12 \lambda^T A = f^T \bigl(AA^T\bigr)^{-1} A
```
```math
x = A^T \bigl(AA^T\bigr)^{-1} f
```
Выражение ``A^T \bigl(AA^T\bigr)^{-1}`` 
-- это правая псевдообратная матрица для ``A``.

В Julia мы можем вычислить `x` как `pinv(A) * f`
или даже просто как
`A \ f`.
"""

# ╔═╡ 4b51eecc-cab2-4f55-87aa-eb3b2e556c60
function get_dr(data)
	r = data["r"]
	p = data["p"]
	theta = data["theta"]

	A, f = make_system(r, p, theta)
	x = A \ f
	return reshape(x, (2, :))'
end

# ╔═╡ 50eeeeb9-dd79-44f8-b0e9-96fef75baf5c
get_dr(data)

# ╔═╡ ac7fce2c-3479-4c59-9ba4-cda33593dbe6
md" Это отличается от предполагаемого ответа:"

# ╔═╡ 5dd22bbb-d4c1-40f5-a006-945149724697
data["dr"]

# ╔═╡ bb520191-f723-4249-ab07-fc1d2a345ee6
md"Вычислим величину ``Ax - f`` для моего решения и для предполагаемого:"

# ╔═╡ 13533bd7-512c-4f8c-9ff5-73933a9e02e9
A, f = make_system(data["r"], data["p"], data["theta"])

# ╔═╡ 3b6ed5b4-e3bd-4c5a-a152-b5f77115fb84
# мое решение: Ax - f ≈ 0
A * vec(get_dr(data)') - f

# ╔═╡ 6aa07ff1-b4cc-4074-a872-a8bad3c80a38
# Предполагаемое решение:
A * vec(data["dr"]') - f

# ╔═╡ 3b3cdbd5-f9e2-468f-bb2c-5376b8da1c97
md"Для моего решения ``Ax-f \approx 0``, для предполгаемого -- нет. Следовательно, проблема не на этапе решения задачи наименьших квадратов, а на этапе составления линейной системы."

# ╔═╡ 7514dc56-d87f-48b2-8e37-55db46873a89
md"Вычислим также, насколько хорошо мой метод решил исходную нелинеаризованную задачу:"

# ╔═╡ dea84ecf-24e1-4072-8600-6d85172428d1
function residual(r, p, theta)
	# Функция, которая вычисляет невязки rᵢⱼ ⋅ rᵢₖ - |rᵢⱼ| * |rᵢₖ| * cos(θ)
	ret = zeros(length(theta))
	for (n, (i, j, k)) ∈ enumerate(eachrow(p))
		r_ij = r[i, :] - r[j, :]
		r_ik = r[i, :] - r[k, :]
		ret[n] = r_ij ⋅ r_ik - norm(r_ij) * norm(r_ik) * cos(theta[n])
	end
	return ret
end

# ╔═╡ 8dfe33e9-242a-416d-9bab-6c2ae54791b8
# исходные точки
residual(data["r"], data["p"], data["theta"])

# ╔═╡ 1bf99814-9c0d-4c56-acd5-ea808c8d6abc
# моё решение
residual(data["r"] + get_dr(data), data["p"], data["theta"])

# ╔═╡ 312f5130-6934-4aee-a37b-717fd1486d78
# правильный ответ
residual(data["r"] + data["dr"], data["p"], data["theta"])

# ╔═╡ 55e81bf4-207c-44b1-9ae8-a7c675826a24
md"Как мы видим, моё решение имеет более маленькую невязку во всех уравнениях, кроме третьего. В третьем уравнении она равна 0.07, что на порядок больше, чем в остальных уравнениях.

Что именно не так с третьем уравнением, понять не удалось."

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
NPZ = "15e1cf62-19b3-5cfa-8e77-841668bca605"

[compat]
NPZ = "~0.4.2"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.2"
manifest_format = "2.0"
project_hash = "f1bb52fd6db9f9a6824a021d35b5a66a09371419"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "78bee250c6826e1cf805a88b7f1e86025275d208"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.46.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "0.5.2+0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "7be5f99f7d15578798f338f5433b6c432ea8037b"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.16.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.0+0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[deps.NPZ]]
deps = ["Compat", "FileIO", "ZipFile"]
git-tree-sha1 = "45f77b87cb9ed5b519f31e1590258930f3b840ee"
uuid = "15e1cf62-19b3-5cfa-8e77-841668bca605"
version = "0.4.2"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.ZipFile]]
deps = ["Libdl", "Printf", "Zlib_jll"]
git-tree-sha1 = "3593e69e469d2111389a9bd06bac1f3d730ac6de"
uuid = "a5390f91-8eb1-5f08-bee0-b1d1ffed6cea"
version = "0.9.4"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.12+3"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"
"""

# ╔═╡ Cell order:
# ╠═0572e29a-5ae4-11ed-2a0d-ebfa7b174b56
# ╠═652d1c8d-d6a8-4424-a136-da3ba9d6a638
# ╟─9d1b650a-01b3-48e9-9328-c78e4170741b
# ╠═6297c7a2-5964-4b8d-a532-19c6c528e65f
# ╟─6ace1ea9-2654-404e-a09b-20e72c287db2
# ╠═4b51eecc-cab2-4f55-87aa-eb3b2e556c60
# ╠═50eeeeb9-dd79-44f8-b0e9-96fef75baf5c
# ╟─ac7fce2c-3479-4c59-9ba4-cda33593dbe6
# ╠═5dd22bbb-d4c1-40f5-a006-945149724697
# ╟─bb520191-f723-4249-ab07-fc1d2a345ee6
# ╠═13533bd7-512c-4f8c-9ff5-73933a9e02e9
# ╠═3b6ed5b4-e3bd-4c5a-a152-b5f77115fb84
# ╠═6aa07ff1-b4cc-4074-a872-a8bad3c80a38
# ╟─3b3cdbd5-f9e2-468f-bb2c-5376b8da1c97
# ╟─7514dc56-d87f-48b2-8e37-55db46873a89
# ╠═dea84ecf-24e1-4072-8600-6d85172428d1
# ╠═8dfe33e9-242a-416d-9bab-6c2ae54791b8
# ╠═1bf99814-9c0d-4c56-acd5-ea808c8d6abc
# ╠═312f5130-6934-4aee-a37b-717fd1486d78
# ╟─55e81bf4-207c-44b1-9ae8-a7c675826a24
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
