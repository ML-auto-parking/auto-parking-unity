using System;
using System.Collections.Generic;
using System.Linq;

namespace AutonomousParking.Common.Extensions
{
    // ExceptionThrowingExtensions 클래스는 입력 인자나 컬렉션에 대해 조건을 확인하고
    // 조건에 맞지 않을 때 예외를 던지는 확장 메서드를 제공합니다.
    
    public static class ExceptionThrowingExtensions
    {
        // 매개변수가 null인 경우 ArgumentNullException을 발생시키는 확장 메서드입니다.
        public static T ThrowExceptionIfArgumentIsNull<T>(this T argument, string argumentName)
        {
            if (argument == null)
                throw new ArgumentNullException(argumentName); // 매개변수 이름을 포함하여 예외를 발생시킵니다.
            return argument; // null이 아니면 매개변수를 그대로 반환합니다.
        }

        // 인덱스가 컬렉션의 범위를 벗어나는 경우 ArgumentOutOfRangeException을 발생시키는 확장 메서드입니다.
        public static int ThrowExceptionIfArgumentOutOfRange<T>(this int argument, string argumentName,
            ICollection<T> collection) =>
            argument.ThrowExceptionIfArgumentOutOfRange(argumentName, collection.FirstIndex(), collection.LastIndex());

        // 매개변수 값이 지정된 범위를 벗어나는 경우 ArgumentOutOfRangeException을 발생시키는 확장 메서드입니다.
        // 이 메서드는 T 타입이 IComparable<T> 인터페이스를 구현해야 합니다.
        public static T ThrowExceptionIfArgumentOutOfRange<T>(this T argument, string argumentName,
            T minValue, T maxValue) where T : IComparable<T>
        {
            if (argument.CompareTo(minValue) < (int)default) // minValue보다 작은 경우
                throw new ArgumentOutOfRangeException(argumentName, $"Value cannot be less than {minValue}.");
            if (argument.CompareTo(maxValue) > (int)default) // maxValue보다 큰 경우
                throw new ArgumentOutOfRangeException(argumentName, $"Value cannot be greater than {maxValue}.");
            return argument; // 지정된 범위 내라면 매개변수를 그대로 반환합니다.
        }

        // 시퀀스에 요소가 없는 경우 InvalidOperationException을 발생시키는 확장 메서드입니다.
        public static IEnumerable<T> ThrowExceptionIfNoElements<T>(this IEnumerable<T> source)
        {
            if (!source.Any()) // 요소가 하나도 없다면
                throw new InvalidOperationException("Sequence contains no elements");
            return source; // 요소가 있으면 시퀀스를 그대로 반환합니다.
        }
    }
}
