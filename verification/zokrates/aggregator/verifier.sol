// This file is MIT Licensed.
//
// Copyright 2017 Christian Reitwiessner
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
pragma solidity ^0.8.0;
library Pairing {
    struct G1Point {
        uint X;
        uint Y;
    }
    // Encoding of field elements is: X[0] * z + X[1]
    struct G2Point {
        uint[2] X;
        uint[2] Y;
    }
    /// @return the generator of G1
    function P1() pure internal returns (G1Point memory) {
        return G1Point(1, 2);
    }
    /// @return the generator of G2
    function P2() pure internal returns (G2Point memory) {
        return G2Point(
            [10857046999023057135944570762232829481370756359578518086990519993285655852781,
             11559732032986387107991004021392285783925812861821192530917403151452391805634],
            [8495653923123431417604973247489272438418190587263600148770280649306958101930,
             4082367875863433681332203403145435568316851327593401208105741076214120093531]
        );
    }
    /// @return the negation of p, i.e. p.addition(p.negate()) should be zero.
    function negate(G1Point memory p) pure internal returns (G1Point memory) {
        // The prime q in the base field F_q for G1
        uint q = 21888242871839275222246405745257275088696311157297823662689037894645226208583;
        if (p.X == 0 && p.Y == 0)
            return G1Point(0, 0);
        return G1Point(p.X, q - (p.Y % q));
    }
    /// @return r the sum of two points of G1
    function addition(G1Point memory p1, G1Point memory p2) internal view returns (G1Point memory r) {
        uint[4] memory input;
        input[0] = p1.X;
        input[1] = p1.Y;
        input[2] = p2.X;
        input[3] = p2.Y;
        bool success;
        assembly {
            success := staticcall(sub(gas(), 2000), 6, input, 0xc0, r, 0x60)
            // Use "invalid" to make gas estimation work
            switch success case 0 { invalid() }
        }
        require(success);
    }


    /// @return r the product of a point on G1 and a scalar, i.e.
    /// p == p.scalar_mul(1) and p.addition(p) == p.scalar_mul(2) for all points p.
    function scalar_mul(G1Point memory p, uint s) internal view returns (G1Point memory r) {
        uint[3] memory input;
        input[0] = p.X;
        input[1] = p.Y;
        input[2] = s;
        bool success;
        assembly {
            success := staticcall(sub(gas(), 2000), 7, input, 0x80, r, 0x60)
            // Use "invalid" to make gas estimation work
            switch success case 0 { invalid() }
        }
        require (success);
    }
    /// @return the result of computing the pairing check
    /// e(p1[0], p2[0]) *  .... * e(p1[n], p2[n]) == 1
    /// For example pairing([P1(), P1().negate()], [P2(), P2()]) should
    /// return true.
    function pairing(G1Point[] memory p1, G2Point[] memory p2) internal view returns (bool) {
        require(p1.length == p2.length);
        uint elements = p1.length;
        uint inputSize = elements * 6;
        uint[] memory input = new uint[](inputSize);
        for (uint i = 0; i < elements; i++)
        {
            input[i * 6 + 0] = p1[i].X;
            input[i * 6 + 1] = p1[i].Y;
            input[i * 6 + 2] = p2[i].X[1];
            input[i * 6 + 3] = p2[i].X[0];
            input[i * 6 + 4] = p2[i].Y[1];
            input[i * 6 + 5] = p2[i].Y[0];
        }
        uint[1] memory out;
        bool success;
        assembly {
            success := staticcall(sub(gas(), 2000), 8, add(input, 0x20), mul(inputSize, 0x20), out, 0x20)
            // Use "invalid" to make gas estimation work
            switch success case 0 { invalid() }
        }
        require(success);
        return out[0] != 0;
    }
    /// Convenience method for a pairing check for two pairs.
    function pairingProd2(G1Point memory a1, G2Point memory a2, G1Point memory b1, G2Point memory b2) internal view returns (bool) {
        G1Point[] memory p1 = new G1Point[](2);
        G2Point[] memory p2 = new G2Point[](2);
        p1[0] = a1;
        p1[1] = b1;
        p2[0] = a2;
        p2[1] = b2;
        return pairing(p1, p2);
    }
    /// Convenience method for a pairing check for three pairs.
    function pairingProd3(
            G1Point memory a1, G2Point memory a2,
            G1Point memory b1, G2Point memory b2,
            G1Point memory c1, G2Point memory c2
    ) internal view returns (bool) {
        G1Point[] memory p1 = new G1Point[](3);
        G2Point[] memory p2 = new G2Point[](3);
        p1[0] = a1;
        p1[1] = b1;
        p1[2] = c1;
        p2[0] = a2;
        p2[1] = b2;
        p2[2] = c2;
        return pairing(p1, p2);
    }
    /// Convenience method for a pairing check for four pairs.
    function pairingProd4(
            G1Point memory a1, G2Point memory a2,
            G1Point memory b1, G2Point memory b2,
            G1Point memory c1, G2Point memory c2,
            G1Point memory d1, G2Point memory d2
    ) internal view returns (bool) {
        G1Point[] memory p1 = new G1Point[](4);
        G2Point[] memory p2 = new G2Point[](4);
        p1[0] = a1;
        p1[1] = b1;
        p1[2] = c1;
        p1[3] = d1;
        p2[0] = a2;
        p2[1] = b2;
        p2[2] = c2;
        p2[3] = d2;
        return pairing(p1, p2);
    }
}

contract Verifier {
    using Pairing for *;
    struct VerifyingKey {
        Pairing.G1Point alpha;
        Pairing.G2Point beta;
        Pairing.G2Point gamma;
        Pairing.G2Point delta;
        Pairing.G1Point[] gamma_abc;
    }
    struct Proof {
        Pairing.G1Point a;
        Pairing.G2Point b;
        Pairing.G1Point c;
    }
    function verifyingKey() pure internal returns (VerifyingKey memory vk) {
        vk.alpha = Pairing.G1Point(uint256(0x29de4e6a28eaf8d24513a3782a841c8a0f952865864ea92427f6868a00ea3f8b), uint256(0x0038563bf7f9a5fc56fb62fe2647389fdd620279413a7b2c8206877870fdb25c));
        vk.beta = Pairing.G2Point([uint256(0x01dc323046b1735579a2e6c92b6f18c67be0b5cf5d9132fae894d58c1c60743f), uint256(0x120c82b5db725b30080fb248030adea1092c3e38a79f58afcabacc7d75a61341)], [uint256(0x28fea95ade134edd3c58d759f5f16d91b5d578ec614f3b0876b76f94a95f2b47), uint256(0x2e648607b7bc5a4504b5dcc70ee9466c1c2b7f6b6fb8e4d58bd5532fca60c27b)]);
        vk.gamma = Pairing.G2Point([uint256(0x12905361ef2b8ed1655da1ff0aa1615b188aacd2e9644b9a461f5bd7a2dd1ebf), uint256(0x09428e31eb7b4d4c26e66e0a4e0a326907cd112dcce433f7ab2618ba0b456fe5)], [uint256(0x0765ce460b8af8480fd8a241dd4a3b703018e45ece406862a06f664bdd366ab1), uint256(0x0d2571d7486fb6f91c178bf9e676aed0e153a6e1589252313ccfc7aa039167ec)]);
        vk.delta = Pairing.G2Point([uint256(0x065f6ed14664a9aae738a923f1042824179808a1036a22866a4aa4b61e8e3f85), uint256(0x0481c1fa4349bfc203965e03b76023977494496a817ae6c30d988b8364a3bb01)], [uint256(0x235a6c168937b2ba93d9ebcdc5400a93587973d208425b3b6c5d53e29867dcb8), uint256(0x255c43af177385422459c45b9ed6b225a1d4fb73fd8d992530eb5a72061252ae)]);
        vk.gamma_abc = new Pairing.G1Point[](11);
        vk.gamma_abc[0] = Pairing.G1Point(uint256(0x150fbd802ca755aa25ca3c0fb6fc9b9f69e7deefdb47bc5e94df37ddfa317605), uint256(0x1828cb47acc0eb0d98d9e55a4f253d318ce2f4e882d8045a8099eb0d81bffe27));
        vk.gamma_abc[1] = Pairing.G1Point(uint256(0x1950fb83d9c3fb4989423d58d77eeb1333eb169335434ab8e74874875b539f2c), uint256(0x09c95ca00850fc59797cc08f5342081c467179ab53fa59cba5256cbf33d99593));
        vk.gamma_abc[2] = Pairing.G1Point(uint256(0x2331c1006701f6837cd2cb2fbc658de9c0999d23d306fe29d3aec71be164dd03), uint256(0x10f03ca99292b9632a1cbb3333307d8f89d38be61edd25202a07844bca0b8bc8));
        vk.gamma_abc[3] = Pairing.G1Point(uint256(0x05996398e49740d32d94b94c2dad9e27dfdd71e06b98fac30afb0b0ff01dad2d), uint256(0x23857dad6fc35f0bd4600cd02f2ec27ef2563f7ea05c52472f2569ad2fcc0ac7));
        vk.gamma_abc[4] = Pairing.G1Point(uint256(0x279442693dacb1dbdcad42148cbc0c76f4d2311581628c4d5ee92e4f95ff137b), uint256(0x1416af31f28abe2d2d1bd41f459b2148862baa1c0cdfb8050179b2eee3f9909c));
        vk.gamma_abc[5] = Pairing.G1Point(uint256(0x28fc8fe114e4a4c337a19e39695003886cab59e8f000ffec15e3b58441213b68), uint256(0x2c1dbbdbfabc58e1f05c38f0ee831ffca3a61e144df7c7ed6d8ad209a5add7a2));
        vk.gamma_abc[6] = Pairing.G1Point(uint256(0x18d3ef6eb3027fc323abd9840c0c3f6fefa20f664978dd51500d8cb30e50373e), uint256(0x0e164b9efb9cbd8d47889d76b9257e5643f1c8897a84ea3eda1918f955012c9c));
        vk.gamma_abc[7] = Pairing.G1Point(uint256(0x0d86011689b0cd52d023ae4fdc873296964bb595248fb4b22509824bdf81b37e), uint256(0x02b399862f4d4cc370be84c1a4ca9e2c9869242030dddeb1ff26f00490b41955));
        vk.gamma_abc[8] = Pairing.G1Point(uint256(0x1b2407de08d601dc34e97920ca15e58febbca0fcbe70715e77ebd4ffdb868f15), uint256(0x3015593ae4cc2bc5f0bd70eb8c6c3b1028aca3ce433dc4915489a18e86aa04a6));
        vk.gamma_abc[9] = Pairing.G1Point(uint256(0x0cc22f56f3753ce9f881bae0fdd87d501452eea278e90a66e9ed2efc7b48fbc3), uint256(0x22c9013cc4c0c582b038dde959ee6f3efc98762b7aa8b3c7de518144f97674f3));
        vk.gamma_abc[10] = Pairing.G1Point(uint256(0x2fef82a00ea4e921cb087d72da38f0508c8ed048a561397b16be1941d9b118c4), uint256(0x16cf21027d04a1e9970ccc07aae03284650dcf92fbe1d8ad42f33e3488720ac3));
    }
    function verify(uint[] memory input, Proof memory proof) internal view returns (uint) {
        uint256 snark_scalar_field = 21888242871839275222246405745257275088548364400416034343698204186575808495617;
        VerifyingKey memory vk = verifyingKey();
        require(input.length + 1 == vk.gamma_abc.length);
        // Compute the linear combination vk_x
        Pairing.G1Point memory vk_x = Pairing.G1Point(0, 0);
        for (uint i = 0; i < input.length; i++) {
            require(input[i] < snark_scalar_field);
            vk_x = Pairing.addition(vk_x, Pairing.scalar_mul(vk.gamma_abc[i + 1], input[i]));
        }
        vk_x = Pairing.addition(vk_x, vk.gamma_abc[0]);
        if(!Pairing.pairingProd4(
             proof.a, proof.b,
             Pairing.negate(vk_x), vk.gamma,
             Pairing.negate(proof.c), vk.delta,
             Pairing.negate(vk.alpha), vk.beta)) return 1;
        return 0;
    }
    function verifyTx(
            Proof memory proof, uint[10] memory input
        ) public view returns (bool r) {
        uint[] memory inputValues = new uint[](10);
        
        for(uint i = 0; i < input.length; i++){
            inputValues[i] = input[i];
        }
        if (verify(inputValues, proof) == 0) {
            return true;
        } else {
            return false;
        }
    }
}
