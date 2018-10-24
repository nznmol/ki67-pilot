package main

import (
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
)

func main() {
	imgPath := os.Getenv("IMG_PATH")
	if imgPath == "" {
		log.Fatal("Must supply image path with env var IMG_PATH")
	}
	dir, absPath := filepath.Split(imgPath)
	absPatchPathFormat := string(absPath[0]) + "_%d_%d_small" + absPath[1:]
	for patchNumber := 0; patchNumber < 225; patchNumber++ {
		verticalOffset := 50 * (patchNumber / 15)
		horizontalOffset := 50 * (patchNumber % 15)
		absPatch := fmt.Sprintf(absPatchPathFormat+"\n", verticalOffset,
			horizontalOffset)
		patchPath := dir + absPatch
		cropCommandFormat := "convert %s -crop 50x50+%d+%d %s"
		cropCommand := fmt.Sprintf(cropCommandFormat, imgPath, verticalOffset,
			horizontalOffset, patchPath)
		cmd := exec.Command("bash", "-c", cropCommand)
		stdoutStderr, err := cmd.CombinedOutput()
		if err != nil {
			log.Fatal(err)
		}
		log.Printf("%s", stdoutStderr)
	}
}
