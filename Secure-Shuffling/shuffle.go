package main

import (
	"crypto/rand"
	_ "fmt"
	"log"
	"shufflemessage/mycrypto"
)

var msgBlocks, numServers, PRG_d int

func msgSim_Camel(msgType, msgBlocks int, numServers int) []byte {
	msg := mycrypto.MakeMsg(msgBlocks, msgType)
	mac, keySeeds := mycrypto.WeirdMac(numServers, msg, false)
	bodyShares := mycrypto.Share(numServers, append(msg, mac...))
	msgToSend_0 := append(bodyShares[0], keySeeds[0]...)
	msgToSend_1 := append(bodyShares[1], keySeeds[1]...)
	msgToSend := append(msgToSend_0, msgToSend_1...)
	if msgBlocks == 2 {
		seed := make([]byte, 16)
		_ = mycrypto.AesPRG(PRG_d, seed)
	}
	return msgToSend
}

func ClientSharingPhase(db0 [][]byte, db1 [][]byte, msgBlocks, batchSize int) {
	shareLength := 16*msgBlocks + 16 + 16
	seed := make([]byte, 16)
	_, err := rand.Read(seed)
	if err != nil {
		log.Println("couldn't generate seed")
		panic(err)
	}
	prelimPerm := mycrypto.GenPerm(batchSize, seed)
	numThreads, chunkSize := mycrypto.PickNumThreads(batchSize)
	blocker := make(chan int)
	for i := 0; i < numThreads; i++ {
		startIndex := i * chunkSize
		endIndex := (i + 1) * chunkSize
		go func(startI, endI, threadNum int) {
			for msgCount := startI; msgCount < endI; msgCount++ {
				clientTransmission := msgSim_Camel(msgCount%26, msgBlocks, numServers)
				copy(db0[prelimPerm[msgCount]][0:shareLength], clientTransmission[0:shareLength])
				copy(db1[prelimPerm[msgCount]][0:shareLength], clientTransmission[shareLength:2*shareLength])
			}
			blocker <- 1
		}(startIndex, endIndex, i)
	}
	for i := 0; i < numThreads; i++ {
		<-blocker
	}
}

func expandDB(db0 [][]byte, db1 [][]byte, msgBlocks int) {
	blocker := make(chan int)
	batchSize := len(db0)
	numThreads, chunkSize := mycrypto.PickNumThreads(batchSize)
	for i := 0; i < numThreads; i++ {
		startIndex := i * chunkSize
		endIndex := (i + 1) * chunkSize
		go func(startI, endI int) {
			for j := startI; j < endI; j++ {
				copy(db0[j][(msgBlocks+1)*16:],
					mycrypto.AesPRG(msgBlocks*16, db0[j][(msgBlocks+1)*16:(msgBlocks+2)*16]))
			}
			for j := startI; j < endI; j++ {
				copy(db1[j][(msgBlocks+1)*16:],
					mycrypto.AesPRG(msgBlocks*16, db1[j][(msgBlocks+1)*16:(msgBlocks+2)*16]))
			}
			blocker <- 1
		}(startIndex, endIndex)
	}

	for i := 0; i < numThreads; i++ {
		<-blocker
	}
}

func flatten(db [][]byte, flatDB []byte) {
	rowLen := len(db[0])
	for i := 0; i < len(db); i++ {
		copy(flatDB[i*rowLen:(i+1)*rowLen], db[i])
	}
}

func mergeFlattenedDBs(flatDBs []byte, numServers, dbSize int) []byte {
	if dbSize%16 != 0 || len(flatDBs) != numServers*dbSize {
		panic("something is wrong with the MergeFlattenedDBs parameters")
	}
	dbs := make([][]byte, numServers)
	for i := 0; i < numServers; i++ {
		dbs[i] = flatDBs[i*dbSize : (i+1)*dbSize]
	}

	return mycrypto.Merge(dbs)
}

func exp_camel(vec_dim, batchsz int) {
	d := vec_dim
	PRG_d = (32 * d) / 128 // 32 bits for each dimension
	msgBlocks = 2          // 1 bit (sign) + 128 bit (seed) for each compressed gradient, we pad 1 bit to 128 bit, i.e., each compressed gradient requires 2 blocks storage
	numServers = 2
	batchSize := batchsz
	blocksPerRow := 2*msgBlocks + 1 // blocksPerRow = Msg (msgBlocks) + Mac (1) + Mac_key (msgBlocks)
	numBeavers := batchSize * msgBlocks
	dbSize := blocksPerRow * batchSize * 16

	db0 := make([][]byte, batchSize)
	db1 := make([][]byte, batchSize)
	for i := 0; i < batchSize; i++ {
		db0[i] = make([]byte, blocksPerRow*16)
		db1[i] = make([]byte, blocksPerRow*16)
	}
	flatDB0 := make([]byte, dbSize)
	flatDB1 := make([]byte, dbSize)

	ClientSharingPhase(db0, db1, msgBlocks, batchSize)

	a_1 := make([]byte, 0)
	b_2 := make([]byte, 0)
	a_prime_2 := make([]byte, 0)
	delta := make([]byte, 0)
	pi1 := make([]int, 0)
	pi2 := make([]int, 0)

	seeds := make([]byte, 80)
	_, err := rand.Read(seeds[:])
	if err != nil {
		log.Println("couldn't generate seed")
		panic(err)
	}

	beaverSeeds := make([][]byte, numServers)
	for i := 0; i < numServers; i++ {
		beaverSeeds[i] = make([]byte, 96)
		_, err := rand.Read(beaverSeeds[i])
		if err != nil {
			panic(err)
		}
	}
	expandDB(db0, db1, msgBlocks)
	pi1 = mycrypto.GenPerm(batchSize, seeds[64:80])
	pi2 = mycrypto.GenPerm(batchSize, seeds[48:64])
	a_1 = mycrypto.AesPRG(dbSize, seeds[0:16])
	b_2 = mycrypto.AesPRG(dbSize, seeds[16:32])
	a_prime_2 = mycrypto.AesPRG(dbSize, seeds[32:48])

	// Shuffle correlation
	pi1_a1 := mycrypto.PermuteDB(a_1, pi1)
	mycrypto.AddOrSub(pi1_a1, a_prime_2, true)
	pi2_tmp := mycrypto.PermuteDB(pi1_a1, pi2)
	mycrypto.AddOrSub(pi2_tmp, b_2, false)
	delta = pi2_tmp

	// Beaver triples
	beaversAShares := make([][]byte, numServers)
	beaversBShares := make([][]byte, numServers)
	for i := 0; i < numServers; i++ {
		beaversAShares[i] = mycrypto.AesPRG(16*numBeavers, beaverSeeds[i][48:64])
		beaversBShares[i] = mycrypto.AesPRG(16*numBeavers, beaverSeeds[i][64:80])
	}
	beaversCShares := mycrypto.GenBeavers(numBeavers, 48, beaverSeeds)

	// Step 1 of Secure Shuffling
	// S2 masks their DB share and sends it to P1
	mycrypto.AddOrSub(flatDB1, a_1, false) // false is for subtraction // z_2 = [x]_2 - a_1

	// first check here
	maskedShares := append(flatDB0, flatDB1...)                             // Step 1: S1 locally computes x-a1
	x_minus_a1 := mergeFlattenedDBs(maskedShares, numServers, len(flatDB0)) // Step 1: S1 locally computes x-a1
	x_minus_a1_shares := mycrypto.Share(numServers, x_minus_a1)             // Step 2: S1 splits x-a1 into 2 shares
	a1_shares := mycrypto.Share(numServers, a_1)                            // Step 3: S3 splits a1 into 2 shares
	mycrypto.AddOrSub(x_minus_a1_shares[0], a1_shares[0], true)             // Step 4: summing
	mycrypto.AddOrSub(x_minus_a1_shares[1], a1_shares[1], true)             // Step 4: summing
	x_to_verify0 := x_minus_a1_shares[0]
	x_to_verify1 := x_minus_a1_shares[1]
	first_check_numThreads, first_check_chunkSize := mycrypto.PickNumThreads(batchSize)
	first_check_hashBlocker := make(chan int)
	//unflatten DB
	for i := 0; i < first_check_numThreads; i++ {
		startI := i * first_check_chunkSize
		endI := (i + 1) * first_check_chunkSize
		go func(startIndex, endIndex int) {
			for j := startIndex; j < endIndex; j++ {
				db0[j] = x_to_verify0[j*blocksPerRow*16 : (j+1)*blocksPerRow*16]
				db1[j] = x_to_verify1[j*blocksPerRow*16 : (j+1)*blocksPerRow*16]
			}
		}(startI, endI)
	}
	first_check_hash0 := make([]byte, 0)
	first_check_hash1 := make([]byte, 0)
	go func() {
		first_check_hash0 = mycrypto.Hash(x_to_verify0)
		first_check_hash1 = mycrypto.Hash(x_to_verify1)
		_, _ = first_check_hash0, first_check_hash1
		first_check_hashBlocker <- 1
	}()
	first_check_maskedStuff_0 := mycrypto.GetMaskedStuff(batchSize, msgBlocks, beaversAShares[0], beaversBShares[0], db0, true)
	first_check_maskedStuff_1 := mycrypto.GetMaskedStuff(batchSize, msgBlocks, beaversAShares[1], beaversBShares[1], db1, true)
	first_check_maskedShares := append(first_check_maskedStuff_0, first_check_maskedStuff_1...)
	first_check_mergedMaskedShares := mergeFlattenedDBs(first_check_maskedShares, numServers, len(first_check_maskedStuff_0))
	first_check_macDiffShares_0 := mycrypto.BeaverProduct(msgBlocks, batchSize, beaversCShares[0], first_check_mergedMaskedShares, db0, true, false)
	first_check_macDiffShares_1 := mycrypto.BeaverProduct(msgBlocks, batchSize, beaversCShares[1], first_check_mergedMaskedShares, db1, false, false)
	first_check_hashedMacDiffShares_0 := mycrypto.Hash(first_check_macDiffShares_0)
	first_check_hashedMacDiffShares_1 := mycrypto.Hash(first_check_macDiffShares_1)
	first_check_allHashedMacDiffShares := append(first_check_hashedMacDiffShares_0, first_check_hashedMacDiffShares_1...)
	first_check_finalMacDiffShares := append(first_check_macDiffShares_0, first_check_macDiffShares_1...)
	if !mycrypto.CheckHashes(first_check_allHashedMacDiffShares, first_check_finalMacDiffShares, len(first_check_macDiffShares_0), 0) && !mycrypto.CheckHashes(first_check_allHashedMacDiffShares, first_check_finalMacDiffShares, len(first_check_macDiffShares_1), 1) {
		panic("******* first check mac hashes did not match")
	} else {
		log.Printf("******* first check mac hashes matched")
	}
	first_check_success := mycrypto.CheckSharesAreZero(batchSize, numServers, first_check_finalMacDiffShares)
	if !first_check_success {
		panic("******* first check blind mac verification failed")
	} else {
		log.Printf("******* first check blind mac verification passed")
	}

	// Step 2 of Secure Shuffling
	// S2 receive the values (z_2) masked with a_1
	mycrypto.AddOrSub(flatDB0, flatDB1, true) // flatDB0 = z_2 + [x]_1
	flatDB0 = mycrypto.PermuteDB(flatDB0, pi1)
	mycrypto.AddOrSub(flatDB0, a_prime_2, false) // flatDB0 is z_1
	z_1 := flatDB0                               // S1 computes z_1, and then sends it to P2
	flatDB0 = b_2                                // S1 set b_2 as the final share

	// second check here
	pi2_z1 := mycrypto.PermuteDB(z_1, pi2)              // Step 1: S2 locally permutes pi_2(z_1)
	pi2_z1_shares := mycrypto.Share(numServers, pi2_z1) // Step 2: S2 splits pi_2(z_1) into 2 shares
	a_1 = mycrypto.AesPRG(dbSize, seeds[0:16])
	pi1_a1 = mycrypto.PermuteDB(a_1, pi1)
	mycrypto.AddOrSub(pi1_a1, a_prime_2, true) // Step 3: S3 splits s3_local into 2 shares
	s3_local := mycrypto.PermuteDB(pi1_a1, pi2)
	s3_local_shares := mycrypto.Share(numServers, s3_local)       // Step 3: S3 splits s3_local into 2 shares
	mycrypto.AddOrSub(pi2_z1_shares[0], s3_local_shares[0], true) // Step 4: secure addition
	mycrypto.AddOrSub(pi2_z1_shares[1], s3_local_shares[1], true) // Step 4: secure addition
	pi_x_to_verify0 := pi2_z1_shares[0]
	pi_x_to_verify1 := pi2_z1_shares[1]
	second_check_numThreads, second_check_chunkSize := mycrypto.PickNumThreads(batchSize)
	second_check_hashBlocker := make(chan int)
	for i := 0; i < second_check_numThreads; i++ {
		startI := i * second_check_chunkSize
		endI := (i + 1) * second_check_chunkSize
		go func(startIndex, endIndex int) {
			for j := startIndex; j < endIndex; j++ {
				db0[j] = pi_x_to_verify0[j*blocksPerRow*16 : (j+1)*blocksPerRow*16]
				db1[j] = pi_x_to_verify1[j*blocksPerRow*16 : (j+1)*blocksPerRow*16]
			}
		}(startI, endI)
	}
	second_check_hash0 := make([]byte, 0)
	second_check_hash1 := make([]byte, 0)
	go func() {
		second_check_hash0 = mycrypto.Hash(x_to_verify0)
		second_check_hash1 = mycrypto.Hash(x_to_verify1)
		_, _ = second_check_hash0, second_check_hash1
		second_check_hashBlocker <- 1
	}()
	second_check_maskedStuff_0 := mycrypto.GetMaskedStuff(batchSize, msgBlocks, beaversAShares[0], beaversBShares[0], db0, true)
	second_check_maskedStuff_1 := mycrypto.GetMaskedStuff(batchSize, msgBlocks, beaversAShares[1], beaversBShares[1], db1, true)
	second_check_maskedShares := append(second_check_maskedStuff_0, second_check_maskedStuff_1...)
	second_check_mergedMaskedShares := mergeFlattenedDBs(second_check_maskedShares, numServers, len(second_check_maskedStuff_0))
	second_check_macDiffShares_0 := mycrypto.BeaverProduct(msgBlocks, batchSize, beaversCShares[0], second_check_mergedMaskedShares, db0, true, false)
	second_check_macDiffShares_1 := mycrypto.BeaverProduct(msgBlocks, batchSize, beaversCShares[1], second_check_mergedMaskedShares, db1, false, false)
	second_check_hashedMacDiffShares_0 := mycrypto.Hash(second_check_macDiffShares_0)
	second_check_hashedMacDiffShares_1 := mycrypto.Hash(second_check_macDiffShares_1)
	second_check_allHashedMacDiffShares := append(second_check_hashedMacDiffShares_0, second_check_hashedMacDiffShares_1...)
	second_check_finalMacDiffShares := append(second_check_macDiffShares_0, second_check_macDiffShares_1...)
	if !mycrypto.CheckHashes(second_check_allHashedMacDiffShares, second_check_finalMacDiffShares, len(second_check_macDiffShares_0), 0) && !mycrypto.CheckHashes(second_check_allHashedMacDiffShares, second_check_finalMacDiffShares, len(second_check_macDiffShares_1), 1) {
		panic("******* second check mac hashes did not match")
	} else {
		log.Printf("******* second check mac hashes matched")
	}
	second_check_success := mycrypto.CheckSharesAreZero(batchSize, numServers, second_check_finalMacDiffShares)
	if !second_check_success {
		panic("******* second check blind mac verification failed")
	} else {
		log.Printf("******* second check blind mac verification passed")
	}

	// Step 3 of Secure Shuffling
	// permute and apply delta
	s_2 := mycrypto.PermuteDB(z_1, pi2)
	mycrypto.AddOrSub(s_2, delta, true)
	flatDB1 = s_2

	// post-shuffle blind MAC verification
	numThreads, chunkSize := mycrypto.PickNumThreads(batchSize)
	hashBlocker := make(chan int)
	for i := 0; i < numThreads; i++ {
		startI := i * chunkSize
		endI := (i + 1) * chunkSize
		go func(startIndex, endIndex int) {
			for j := startIndex; j < endIndex; j++ {
				db0[j] = flatDB0[j*blocksPerRow*16 : (j+1)*blocksPerRow*16]
				db1[j] = flatDB1[j*blocksPerRow*16 : (j+1)*blocksPerRow*16]
			}
		}(startI, endI)
	}
	hash0 := make([]byte, 0)
	hash1 := make([]byte, 0)
	go func() {
		hash0 = mycrypto.Hash(flatDB0)
		hash1 = mycrypto.Hash(flatDB1)
		_, _ = hash0, hash1
		hashBlocker <- 1
	}()
	maskedStuff_0 := mycrypto.GetMaskedStuff(batchSize, msgBlocks, beaversAShares[0], beaversBShares[0], db0, true)
	maskedStuff_1 := mycrypto.GetMaskedStuff(batchSize, msgBlocks, beaversAShares[1], beaversBShares[1], db1, true)
	maskedShares = append(maskedStuff_0, maskedStuff_1...)
	mergedMaskedShares := mergeFlattenedDBs(maskedShares, numServers, len(maskedStuff_0))
	macDiffShares_0 := mycrypto.BeaverProduct(msgBlocks, batchSize, beaversCShares[0], mergedMaskedShares, db0, true, false)
	macDiffShares_1 := mycrypto.BeaverProduct(msgBlocks, batchSize, beaversCShares[1], mergedMaskedShares, db1, false, false)
	hashedMacDiffShares_0 := mycrypto.Hash(macDiffShares_0)
	hashedMacDiffShares_1 := mycrypto.Hash(macDiffShares_1)
	allHashedMacDiffShares := append(hashedMacDiffShares_0, hashedMacDiffShares_1...)
	finalMacDiffShares := append(macDiffShares_0, macDiffShares_1...)
	if !mycrypto.CheckHashes(allHashedMacDiffShares, finalMacDiffShares, len(macDiffShares_0), 0) && !mycrypto.CheckHashes(allHashedMacDiffShares, finalMacDiffShares, len(macDiffShares_1), 1) {
		panic("******* post shuffle mac hashes did not match")
	} else {
		log.Printf("******* post shuffle mac hashes matched")
	}
	success := mycrypto.CheckSharesAreZero(batchSize, numServers, finalMacDiffShares)
	if !success {
		panic("******* post shuffle blind mac verification failed")
	} else {
		log.Printf("******* post shuffle blind mac verification passed")
	}
	sharedFlattenedDBs := append(flatDB0, flatDB1...)
	_ = mergeFlattenedDBs(sharedFlattenedDBs, numServers, len(flatDB0))
	if msgBlocks == 2 {
		seed := make([]byte, 16)
		_ = mycrypto.AesPRG(PRG_d, seed)
	}
	log.Println("Batch = ", batchsz)
	log.Println("Vector dim = ", vec_dim)
}

func main() {
	exp_camel(9594, 3200)
}
