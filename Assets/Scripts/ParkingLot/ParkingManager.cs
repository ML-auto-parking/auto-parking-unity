using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ParkingManager : MonoBehaviour
{
    public Vector3[] parkingSpacesPositions; // 주차 공간 위치
    public GameObject[] carPrefabs; // 차량 프리팹 할당
    public int parallelParkingCount; // 평행 주차 공간 개수

    void Start()
    {
        ResetParkingLot();
    }

    void ResetParkingLot()
    {
        int firstEmptySpace = Random.Range(0, parallelParkingCount);
        int secondEmptySpace = Random.Range(parallelParkingCount, parkingSpacesPositions.Length);

        while (firstEmptySpace == secondEmptySpace)
        {
            secondEmptySpace = Random.Range(0, parkingSpacesPositions.Length);
        }

        for (int i = 0; i < parkingSpacesPositions.Length; i++)
        {
            if (i == firstEmptySpace || i == secondEmptySpace)
            {
                continue;
            }

            Quaternion rotation = Quaternion.identity;

            // 평행 주차 공간에 차량 배치 시 90도 또는 270도 회전
            if (i < parallelParkingCount)
            {
                float angle = Random.Range(0, 2) * 180 + 90; // 90도 또는 270도
                rotation = Quaternion.Euler(0, angle, 0);
            }

            Instantiate(carPrefabs[Random.Range(0, carPrefabs.Length)], parkingSpacesPositions[i], rotation);
        }
    }
}
