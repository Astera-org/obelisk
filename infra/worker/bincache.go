package main

//map of binIDs to their names and versions

type BinInfo struct {
	Name    string
	Version string
}

type BinCache struct {
	cache map[int]*BinInfo
}

func (bc *BinCache) GetBinInfo(binID int) *BinInfo {
	if bc.cache[binID] == nil {
		// fectch from jobczar

		return nil
	}
	return bc.cache[binID]
}
