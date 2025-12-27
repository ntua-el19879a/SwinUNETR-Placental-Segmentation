def long_sum(arr, K):
	left = 0
	best = 0
	s = 0
	for right, num in enumerate(arr):
		while s + num > K and left <= right:
			s -= arr[left]
			left += 1
		s += num
		best = max(best,right - left + 1)
	return best
